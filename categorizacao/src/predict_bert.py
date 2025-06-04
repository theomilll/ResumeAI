import logging
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_label_encoder(encoder_path):
    """
    Load label encoder from CSV file.
    """
    label_encoder = {}
    idx_to_label = {}
    
    with open(encoder_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            if line.strip():
                category, idx = line.strip().split(',')
                label_encoder[category] = int(idx)
                idx_to_label[int(idx)] = category
    
    return label_encoder, idx_to_label

class BERTSummaryClassifier:
    """BERT-based classifier for meeting summaries."""
    
    def __init__(self, model_dir):
        """
        Initialize BERT classifier.
        
        Args:
            model_dir: Directory containing the fine-tuned BERT model
        """
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        
        # Force CPU usage for MPS compatibility
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"BERT model loaded from {model_dir}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise RuntimeError(f"Failed to load BERT model: {e}")
        
        # Find and load label encoder
        potential_paths = [
            os.path.join(model_dir, 'label_encoder.csv'),
            os.path.join(os.path.dirname(model_dir), 'label_encoder.csv'),
            './models/label_encoder.csv'
        ]
        
        encoder_path = None
        for path in potential_paths:
            if os.path.exists(path):
                encoder_path = path
                break
                
        if encoder_path:
            try:
                self.label_encoder, self.idx_to_label = load_label_encoder(encoder_path)
                logger.info(f"Label encoder loaded from {encoder_path}")
                logger.info(f"Available categories: {list(self.label_encoder.keys())}")
            except Exception as e:
                logger.error(f"Failed to load label encoder: {e}")
                raise RuntimeError(f"Failed to load label encoder: {e}")
        else:
            logger.error(f"Label encoder not found in any expected location")
            raise FileNotFoundError(f"Label encoder not found. Searched in: {', '.join(potential_paths)}")
    
    def classify(self, text, return_confidence=False):
        """
        Classify a text summary using BERT.
        
        Args:
            text: Text to classify
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predicted category and optionally confidence scores
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Prepare input
        try:
            encoding = self.tokenizer(
                text,
                max_length=256,  # Use same as training
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Create dictionary of category -> confidence
                confidence = {}
                for idx, prob in enumerate(probabilities):
                    if idx in self.idx_to_label:
                        category = self.idx_to_label[idx]
                        confidence[category] = prob.item()
                
                # Get predicted class
                predicted_idx = torch.argmax(logits, dim=1).item()
                predicted_category = self.idx_to_label[predicted_idx]
                
                if return_confidence:
                    return predicted_category, confidence
                
                return predicted_category
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def classify_batch(self, texts):
        """
        Classify multiple texts in a batch.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted categories
        """
        predictions = []
        for text in texts:
            predictions.append(self.classify(text))
        return predictions

def classify_summary(text, model_dir='./models/bert_enhanced'):
    """
    Classify a summary using the enhanced BERT model.
    
    Args:
        text: Text to classify
        model_dir: Directory containing the model
        
    Returns:
        Predicted category
    """
    try:
        classifier = BERTSummaryClassifier(model_dir)
        return classifier.classify(text)
    except Exception as e:
        logger.error(f"Error in classify_summary: {e}")
        raise

def classify_with_confidence(text, model_dir='./models/bert_enhanced'):
    """
    Classify a summary and return confidence scores.
    
    Args:
        text: Text to classify
        model_dir: Directory containing the model
        
    Returns:
        Tuple of (predicted category, confidence scores)
    """
    try:
        classifier = BERTSummaryClassifier(model_dir)
        return classifier.classify(text, return_confidence=True)
    except Exception as e:
        logger.error(f"Error in classify_with_confidence: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify a meeting summary using BERT.")
    parser.add_argument('text', type=str, help='Text to classify')
    parser.add_argument('--model_dir', type=str, default='./models/bert_enhanced',
                       help='Directory containing the BERT model')
    parser.add_argument('--show_confidence', action='store_true',
                       help='Show confidence scores for all categories')
    args = parser.parse_args()

    try:
        if args.show_confidence:
            category, confidence = classify_with_confidence(args.text, args.model_dir)
            print(f"Predicted category: {category}")
            print("\nConfidence scores:")
            for cat, conf in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {conf:.3f}")
        else:
            category = classify_summary(args.text, args.model_dir)
            print(f"Category: {category}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)