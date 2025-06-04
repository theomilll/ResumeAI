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
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from collections import Counter
import logging

# Config
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'  # BERT pré-treinado para português
LABEL_ENCODER_PATH = './models/label_encoder.csv'
MODEL_OUTPUT_DIR = './models/bert_improved'
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

# Data Augmentation
class DataAugmenter:
    def __init__(self, language='pt'):
        # Initialize augmenters for Portuguese
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='por')
        self.back_translation_aug = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-pt-en',
            to_model_name='Helsinki-NLP/opus-mt-en-pt'
        )
        
    def augment_text(self, text, method='synonym'):
        """Augment text using specified method"""
        try:
            if method == 'synonym':
                return self.synonym_aug.augment(text)
            elif method == 'backtranslation':
                return self.back_translation_aug.augment(text)
            else:
                return text
        except:
            # If augmentation fails, return original text
            return text

# Enhanced Dataset with data augmentation
class MeetingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False, augmenter=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmenter = augmenter
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply augmentation during training
        if self.augment and self.augmenter and random.random() < 0.3:  # 30% chance
            method = random.choice(['synonym', 'backtranslation'])
            text = self.augmenter.augment_text(text, method)
            
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Improved BERT fine-tuning for meeting summary classification')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_len', type=int, default=MAX_LEN)
    parser.add_argument('--output_dir', type=str, default=MODEL_OUTPUT_DIR)
    parser.add_argument('--label_column', type=str, default='category', help='Name of the label column')
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data(args.data_path)
    
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found")
        
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
        test_size=0.15, 
        stratify=df['label_idx'], 
        random_state=SEED
    )
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Initialize tokenizer and augmenter
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    augmenter = DataAugmenter() if args.use_augmentation else None
    
    # Create datasets
    train_dataset = MeetingDataset(
        train_df['text'].tolist(), 
        train_df['label_idx'].tolist(), 
        tokenizer, 
        args.max_len,
        augment=args.use_augmentation,
        augmenter=augmenter
    )
    val_dataset = MeetingDataset(
        val_df['text'].tolist(), 
        val_df['label_idx'].tolist(), 
        tokenizer, 
        args.max_len
    )
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label2idx),
        hidden_dropout_prob=0.2,  # Add dropout for regularization
        attention_probs_dropout_prob=0.2
    )
    
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
        report_to=['tensorboard'],
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        remove_unused_columns=False,
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
    
    # Save predictions for analysis
    val_df['predicted_label'] = [idx2label[pred] for pred in y_pred]
    val_df['correct'] = val_df[args.label_column] == val_df['predicted_label']
    
    misclassified_path = os.path.join(args.output_dir, 'misclassified_examples.csv')
    val_df[~val_df['correct']].to_csv(misclassified_path, index=False)
    logger.info(f"Misclassified examples saved to {misclassified_path}")

if __name__ == '__main__':
    main()