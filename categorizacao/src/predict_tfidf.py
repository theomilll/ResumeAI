import argparse

import joblib


def load_label_encoder(encoder_path):
    idx2label = {}
    with open(encoder_path, 'r', encoding='utf-8') as f:
        _ = f.readline()
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            category, idx = line.split(',', 1)
            try:
                idx2label[int(idx)] = category
            except ValueError:
                continue
    return idx2label


def main():
    parser = argparse.ArgumentParser(
        description='CLI for TF-IDF + LogisticRegression meeting classifier')
    parser.add_argument('text', type=str, help='Meeting summary to classify')
    parser.add_argument('--model_path', type=str, default='models/tfidf/tfidf_pipeline.joblib',
                        help='Path to saved TF-IDF pipeline')
    parser.add_argument('--encoder_path', type=str, default='models/tfidf/label_encoder.csv',
                        help='Path to label encoder CSV')
    args = parser.parse_args()

    try:
        pipeline = joblib.load(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    idx2label = load_label_encoder(args.encoder_path)
    if not idx2label:
        print(f"No labels found in encoder at {args.encoder_path}")
        exit(1)

    try:
        pred_idx = pipeline.predict([args.text])[0]
        category = idx2label.get(pred_idx)
        if category is None:
            print(f"Prediction index {pred_idx} not found in encoder mapping")
            exit(1)
        print(f"Category: {category}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        exit(1)


if __name__ == '__main__':
    main()
