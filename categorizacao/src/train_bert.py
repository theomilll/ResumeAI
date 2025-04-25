import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Config
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'  # BERT pré-treinado para português
LABEL_ENCODER_PATH = './models/label_encoder.csv'
MODEL_OUTPUT_DIR = './models/bert_finetuned'
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
MAX_LEN = 128

# Set seed for reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Dataset
class MeetingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
    # Espera colunas: 'text', 'label'
    assert 'text' in df.columns and 'label' in df.columns
    return df

def save_label_encoder(label2idx, path=LABEL_ENCODER_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('category,idx\n')
        for label, idx in label2idx.items():
            f.write(f'{label},{idx}\n')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune BERT for meeting summary classification (Portuguese)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file (columns: text,category)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_len', type=int, default=MAX_LEN)
    parser.add_argument('--output_dir', type=str, default=MODEL_OUTPUT_DIR)
    parser.add_argument('--label_column', type=str, default='category', help='Name of the label/category column in CSV')
    args = parser.parse_args()

    df = load_data(args.data_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in CSV. Available columns: {list(df.columns)}")
    labels = sorted(df[args.label_column].unique())
    label2idx = {label: i for i, label in enumerate(labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    df['label_idx'] = df[args.label_column].map(label2idx)
    save_label_encoder(label2idx)

    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label_idx'], random_state=SEED)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MeetingDataset(train_df['text'].tolist(), train_df['label_idx'].tolist(), tokenizer, args.max_len)
    val_dataset = MeetingDataset(val_df['text'].tolist(), val_df['label_idx'].tolist(), tokenizer, args.max_len)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label2idx))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=SEED,
        save_total_limit=2,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        report = classification_report(labels, preds, target_names=[idx2label[i] for i in range(len(idx2label))], output_dict=True)
        return {
            'accuracy': report['accuracy'],
            'f1_macro': report['macro avg']['f1-score'],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print('Training complete. Model and tokenizer saved.')

    # Evaluation report
    preds_output = trainer.predict(val_dataset)
    y_true = val_df['label_idx'].tolist()
    y_pred = np.argmax(preds_output.predictions, axis=1)
    print('Classification report (validation set):')
    print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()
