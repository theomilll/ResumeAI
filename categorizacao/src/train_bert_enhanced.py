import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import logging

# Config
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'  # BERT pré-treinado para português
LABEL_ENCODER_PATH = './models/label_encoder.csv'
MODEL_OUTPUT_DIR = './models/bert_enhanced'
SEED = 42
BATCH_SIZE = 16  # Increased from 8
EPOCHS = 20  # Increased from 5, but we'll use early stopping
MAX_LEN = 256  # Increased from 128 for better context
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
WEIGHT_DECAY = 0.01

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Force CPU usage for MPS compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
device = torch.device('cpu')

# Simple text augmentation without external libraries
class SimpleAugmenter:
    def __init__(self):
        self.augmentation_methods = [
            self.random_swap,
            self.random_deletion,
            self.synonym_replacement_simple
        ]
    
    def random_swap(self, text, n=2):
        """Randomly swap n words in the text"""
        words = text.split()
        if len(words) < 4:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # If all words were deleted, return a random word
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def synonym_replacement_simple(self, text):
        """Simple synonym replacement for Portuguese"""
        # Dictionary of common synonyms in Portuguese for business context
        synonyms = {
            'reunião': ['encontro', 'sessão', 'assembleia'],
            'projeto': ['iniciativa', 'programa', 'empreendimento'],
            'equipe': ['time', 'grupo', 'equipa'],
            'desenvolvimento': ['evolução', 'progresso', 'crescimento'],
            'análise': ['estudo', 'avaliação', 'exame'],
            'resultado': ['desfecho', 'conclusão', 'consequência'],
            'cliente': ['consumidor', 'comprador', 'usuário'],
            'empresa': ['companhia', 'organização', 'corporação'],
            'problema': ['questão', 'dificuldade', 'desafio'],
            'solução': ['resolução', 'resposta', 'saída']
        }
        
        words = text.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in synonyms and random.random() < 0.3:
                synonym = random.choice(synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def augment(self, text):
        """Apply a random augmentation method"""
        if random.random() < 0.5:  # 50% chance to augment
            method = random.choice(self.augmentation_methods)
            return method(text)
        return text

# Enhanced Dataset with simple augmentation
class MeetingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmenter = SimpleAugmenter() if augment else None
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply augmentation during training
        if self.augment and self.augmenter:
            text = self.augmenter.augment(text)
            
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    df = pd.read_csv(data_path)
    # Expects columns: 'text', 'category' 
    if 'category' in df.columns:
        df['label'] = df['category']
    assert 'text' in df.columns and 'label' in df.columns
    return df

def save_label_encoder(label2idx, path=LABEL_ENCODER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('category,idx\n')
        for label, idx in label2idx.items():
            f.write(f'{label},{idx}\n')

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def analyze_data_distribution(df, label_column):
    """Analyze and print data distribution"""
    print("\n=== Data Distribution Analysis ===")
    label_counts = df[label_column].value_counts()
    total = len(df)
    
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"{label}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nTotal samples: {total}")
    print(f"Number of classes: {len(label_counts)}")
    
    # Calculate imbalance ratio
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    return label_counts

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced BERT fine-tuning for meeting summary classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_len', type=int, default=MAX_LEN)
    parser.add_argument('--output_dir', type=str, default=MODEL_OUTPUT_DIR)
    parser.add_argument('--label_column', type=str, default='category', help='Name of the label column')
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--test_size', type=float, default=0.15, help='Validation set size')
    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data(args.data_path)
    
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found")
    
    # Analyze data distribution
    label_counts = analyze_data_distribution(df, args.label_column)
        
    # Handle label encoding
    labels = sorted(df[args.label_column].unique())
    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    df['label_idx'] = df[args.label_column].map(label2idx)
    save_label_encoder(label2idx)
    
    # Log class distribution
    class_counts = Counter(df['label_idx'])
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(range(len(labels)))),
        y=df['label_idx'].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")
    
    # Split data with stratification
    train_df, val_df = train_test_split(
        df, 
        test_size=args.test_size, 
        stratify=df['label_idx'], 
        random_state=SEED
    )
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = MeetingDataset(
        train_df['text'].tolist(), 
        train_df['label_idx'].tolist(), 
        tokenizer, 
        args.max_len,
        augment=args.use_augmentation
    )
    val_dataset = MeetingDataset(
        val_df['text'].tolist(), 
        val_df['label_idx'].tolist(), 
        tokenizer, 
        args.max_len,
        augment=False  # No augmentation for validation
    )
    
    # Initialize model with dropout for regularization
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label2idx),
        hidden_dropout_prob=0.2,  # Add dropout for regularization
        attention_probs_dropout_prob=0.2
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments with improvements
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1_macro',
        greater_is_better=True,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        save_total_limit=3,
        report_to=[],  # Disable tensorboard for now
        fp16=False,  # Disable mixed precision for MPS
        dataloader_num_workers=0,  # Set to 0 for MPS compatibility
        remove_unused_columns=False,
        push_to_hub=False,
        optim="adamw_torch",  # Use PyTorch AdamW optimizer
        use_mps_device=False,  # Disable MPS for now
        no_cuda=True,  # Force CPU usage
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        
        # Calculate per-class metrics
        report = classification_report(
            labels, 
            preds, 
            target_names=[idx2label[i] for i in range(len(idx2label))], 
            output_dict=True,
            zero_division=0
        )
        
        # Calculate macro F1 score for better evaluation of imbalanced classes
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        
        return {
            'accuracy': report['accuracy'],
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
        }
    
    # Initialize trainer with custom class
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f'Training complete. Model saved to {args.output_dir}')
    
    # Final evaluation
    logger.info("Running final evaluation...")
    preds_output = trainer.predict(val_dataset)
    y_true = val_df['label_idx'].tolist()
    y_pred = np.argmax(preds_output.predictions, axis=1)
    
    print('\n=== Final Classification Report ===')
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=[idx2label[i] for i in range(len(idx2label))],
        digits=3
    ))
    
    print('\n=== Confusion Matrix ===')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Print confusion matrix with labels
    print("\nConfusion Matrix with Labels:")
    print("Predicted ->", end="")
    for i in range(len(idx2label)):
        print(f"\t{idx2label[i][:10]}", end="")
    print()
    for i in range(len(idx2label)):
        print(f"{idx2label[i][:15]}", end="")
        for j in range(len(idx2label)):
            print(f"\t{cm[i][j]}", end="")
        print()
    
    # Save predictions for analysis
    val_df['predicted_label'] = [idx2label[pred] for pred in y_pred]
    val_df['correct'] = val_df[args.label_column] == val_df['predicted_label']
    
    # Save misclassified examples
    misclassified_path = os.path.join(args.output_dir, 'misclassified_examples.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    val_df[~val_df['correct']][['text', args.label_column, 'predicted_label']].to_csv(
        misclassified_path, 
        index=False
    )
    logger.info(f"Misclassified examples saved to {misclassified_path}")
    
    # Calculate and save per-class accuracy
    per_class_accuracy = {}
    for label in labels:
        label_df = val_df[val_df[args.label_column] == label]
        if len(label_df) > 0:
            accuracy = (label_df['correct'].sum() / len(label_df)) * 100
            per_class_accuracy[label] = {
                'accuracy': accuracy,
                'total_samples': len(label_df),
                'correct_predictions': label_df['correct'].sum()
            }
    
    print("\n=== Per-Class Accuracy ===")
    for label, metrics in per_class_accuracy.items():
        print(f"{label}: {metrics['accuracy']:.1f}% ({metrics['correct_predictions']}/{metrics['total_samples']})")

if __name__ == '__main__':
    main()