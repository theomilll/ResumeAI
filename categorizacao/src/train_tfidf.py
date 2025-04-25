import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def save_label_encoder(label2idx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('category,idx\n')
        for label, idx in label2idx.items():
            f.write(f'{label},{idx}\n')


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression classifier for meeting summaries")
    parser.add_argument('--data_path', type=str, required=True, help='CSV file with columns text and category')
    parser.add_argument('--label_column', type=str, default='category', help='Name of category column')
    parser.add_argument('--test_size', type=float, default=0.15, help='Fraction for validation split')
    parser.add_argument('--output_dir', type=str, default='./models/tfidf', help='Dir to save model and encoder')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    if args.label_column not in df.columns or 'text' not in df.columns:
        raise ValueError(f"CSV must contain 'text' and '{args.label_column}' columns. Found: {list(df.columns)}")

    # Encode labels
    labels = sorted(df[args.label_column].unique())
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    df['label_idx'] = df[args.label_column].map(label2idx)

    # Split data
    texts = df['text'].tolist()
    y = df['label_idx'].tolist()
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2)) ),
        ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000))
    ])

    print("Training TF-IDF + LogisticRegression model...")
    pipeline.fit(X_train, y_train)

    # Validation
    preds = pipeline.predict(X_val)
    print("Validation classification report:")
    print(classification_report(y_val, preds, target_names=labels))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, preds))

    # Save model and encoder
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'tfidf_pipeline.joblib')
    enc_path = os.path.join(args.output_dir, 'label_encoder.csv')
    joblib.dump(pipeline, model_path)
    save_label_encoder(label2idx, enc_path)

    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {enc_path}")


if __name__ == '__main__':
    main()
