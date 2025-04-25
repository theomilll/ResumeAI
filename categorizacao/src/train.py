import argparse
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data_preprocessing import (load_and_preprocess_data, prepare_datasets,
                                save_label_encoder, verify_data_format)
from evaluate import compute_metrics
from model import MultiHeadTextClassifier, load_bert_model, save_model


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(train_dataset, val_dataset, model, device, num_epochs=5, batch_size=16, learning_rate=2e-5, 
                model_dir='../models', class_weights=None):
    """Train the text classification model with weighted sampling and learning rate scheduling."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create class-balanced sampler for training data
    if class_weights is not None:
        # Get labels for each training example - with a safe approach
        try:
            # First try direct access to labels
            labels = train_dataset.labels
        except:
            # If that fails, iterate through the dataset
            try:
                labels = []
                for i in range(len(train_dataset)):
                    sample = train_dataset[i]
                    label = sample['label']
                    if hasattr(label, 'item'):
                        labels.append(label.item())
                    else:
                        labels.append(int(label))
            except Exception as e:
                print(f"Warning: Could not extract labels for weighted sampling: {e}")
                print("Using uniform sampling instead")
                class_weights = None
                
        if class_weights is not None:
            # Calculate weights for each sample based on its class
            sample_weights = [class_weights[label] for label in labels]
            
            # Create sampler
            sampler = WeightedRandomSampler(weights=sample_weights, 
                                           num_samples=len(train_dataset), 
                                           replacement=True)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            print("Using weighted random sampling for training")
    
    if class_weights is None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Add a learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_f1 = 0.0
    total_steps = num_epochs * len(train_loader)
    
    print("\n=== Starting Training ===")
    print(f"Total training steps: {total_steps}")
    
    # Track metrics
    epoch_losses = []
    val_metrics = []
    class_predictions = {}
    
    overall_progress = tqdm(total=total_steps, desc="Overall Progress")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch}/{num_epochs} ({epoch/num_epochs*100:.1f}% complete)")
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            overall_progress.update(1)
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_losses.append(avg_loss)
        
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Validation
        print("Running validation...")
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_preds = []
        all_labels = []
        pred_distribution = Counter()
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batch_count += 1
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                
                # Count predictions by class
                for pred in preds:
                    pred_distribution[pred.item()] += 1
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Store prediction distribution for this epoch
        class_predictions[epoch] = dict(pred_distribution)
        
        val_metrics.append({
            'epoch': epoch,
            'loss': avg_val_loss,
            'accuracy': val_accuracy,
            'f1': val_f1,
            'distribution': dict(pred_distribution)
        })
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Prediction distribution: {dict(pred_distribution)}")
        
        if epoch % 2 == 1 or epoch == num_epochs:  # Save every odd epoch and the final epoch
            checkpoint_dir = os.path.join(model_dir, f"checkpoint-epoch-{epoch}")
            # Create directory
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_dir}")
        
        # Update learning rate
        scheduler.step()
        
        # If this is the best model so far based on F1 score, save it
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_dir = os.path.join(model_dir, "best_model")
            # Create directory
            os.makedirs(best_model_dir, exist_ok=True)
            best_model_path = os.path.join(best_model_dir, 'model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_dir}")
    
    overall_progress.close()
    print("\n=== Training Complete ===")
    
    # Print class prediction history
    print("\nClass prediction distribution by epoch:")
    for epoch, dist in class_predictions.items():
        print(f"Epoch {epoch}: {dist}")
    
    # Save the final model
    final_model_dir = os.path.join(model_dir, "final")
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, 'model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_dir}")
    
    # Also save config data for later loading
    model_config = {
        'vocab_size': model.embedding.num_embeddings,
        'embed_dim': model.embedding.embedding_dim,
        'hidden_dim': model.lstm.hidden_size,
        'num_classes': len(model.class_heads)  # Use class_heads instead of fc2.out_features
    }
    
    import json
    with open(os.path.join(final_model_dir, 'config.json'), 'w') as f:
        json.dump(model_config, f)
    
    # Also save as best model if none was saved
    if not os.path.exists(os.path.join(model_dir, "best_model")):
        best_model_dir = os.path.join(model_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_path = os.path.join(best_model_dir, 'model.pt')
        torch.save(model.state_dict(), best_model_path)
        print(f"Final model saved to: {best_model_dir}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a text classification model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased', 
                       help='Pre-trained model to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for balanced training')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Verify data format
    print("\nVerifying data format...")
    if not verify_data_format(args.data_path):
        print("Data format verification failed. Please check your data file.")
        return
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    train_texts, val_texts, train_labels, val_labels, label_encoder = load_and_preprocess_data(
        args.data_path, test_size=0.2, random_state=args.seed
    )
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset = prepare_datasets(
        train_texts, val_texts, train_labels, val_labels, args.pretrained_model
    )
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        # Get class distribution
        class_counts = Counter(train_labels)
        # Calculate inverse frequency
        total_samples = len(train_labels)
        class_weights = {class_idx: total_samples / (len(class_counts) * count) 
                         for class_idx, count in class_counts.items()}
        print(f"\nUsing class weights: {class_weights}")
    
    # Load model
    print("\nLoading model...")
    model = MultiHeadTextClassifier(
        num_classes=len(label_encoder)
    )
    model.to(device)
    
    # Train model
    print("\nTraining model...")
    trained_model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_dir=args.model_dir,
        class_weights=class_weights
    )
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 