import os

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(predictions, labels, class_names, output_dir='./results'):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Model predictions
        labels: True labels
        class_names: Names of the classes
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

def evaluate_model(model, dataloader, device, label_encoder=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: BERT model
        dataloader: Data loader for evaluation
        device: Device to use (cpu/cuda)
        label_encoder: Label encoder dictionary
        
    Returns:
        Dictionary of metrics and predictions
    """
    import torch
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    
    # Convert indices to category names if label_encoder provided
    if label_encoder:
        idx_to_category = {idx: category for category, idx in label_encoder.items()}
        pred_categories = [idx_to_category[idx] for idx in all_preds]
        true_categories = [idx_to_category[idx] for idx in all_labels]
    else:
        pred_categories = all_preds
        true_categories = all_labels
    
    return {
        'metrics': metrics,
        'predictions': pred_categories,
        'true_labels': true_categories
    }

def load_label_encoder(encoder_path):
    """
    Load label encoder from file.
    
    Args:
        encoder_path: Path to label encoder CSV
        
    Returns:
        Label encoder dictionary
    """
    encoder_df = pd.read_csv(encoder_path)
    
    # Create mapping from index to category
    idx_to_category = {row['idx']: row['category'] for _, row in encoder_df.iterrows()}
    
    # Create mapping from category to index
    category_to_idx = {category: idx for idx, category in idx_to_category.items()}
    
    return category_to_idx 