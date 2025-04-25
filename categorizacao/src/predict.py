import logging
import os
import torch
from model import load_model

# Minimal load_label_encoder for inference (copied here to avoid importing evaluate.py or pandas)
def load_label_encoder(encoder_path):
    """
    Load label encoder from file.
    Args:
        encoder_path: Path to label encoder CSV
    Returns:
        Label encoder dictionary
    """
    label_encoder = {}
    col_order = None
    with open(encoder_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            # Detect header and column order
            if 'category' in line.lower() and 'idx' in line.lower():
                parts = [x.strip().lower() for x in line.split(',')]
                if parts[0] == 'category' and parts[1] == 'idx':
                    col_order = 'category_first'
                elif parts[0] == 'idx' and parts[1] == 'category':
                    col_order = 'idx_first'
                continue
            try:
                a, b = line.split(',', 1)
                if col_order == 'category_first':
                    label_encoder[a] = int(b)
                elif col_order == 'idx_first':
                    label_encoder[b] = int(a)
                else:
                    # Try to guess by type
                    if a.isdigit():
                        label_encoder[b] = int(a)
                    elif b.isdigit():
                        label_encoder[a] = int(b)
            except Exception:
                continue
    return label_encoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryClassifier:
    """Classifier for meeting summaries."""
    
    def __init__(self, model_dir):
        """
        Initialize classifier.
        
        Args:
            model_dir: Directory containing the model and label encoder
        """
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.model, self.tokenizer = load_model(model_dir)
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Find and load label encoder - look in several possible locations
        potential_paths = [
            os.path.join(model_dir, 'label_encoder.csv'),                 # In the model directory itself
            os.path.join(os.path.dirname(model_dir), 'label_encoder.csv'),  # In the parent directory
            os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'label_encoder.csv')  # In grandparent directory
        ]
        
        encoder_path = None
        for path in potential_paths:
            if os.path.exists(path):
                encoder_path = path
                break
                
        if encoder_path:
            try:
                self.label_encoder = load_label_encoder(encoder_path)
                # Create reverse mapping
                self.idx_to_category = {idx: category for category, idx in self.label_encoder.items()}
                logger.info(f"Label encoder loaded from {encoder_path}")
                logger.info(f"Available categories: {list(self.label_encoder.keys())}")
            except Exception as e:
                logger.error(f"Failed to load label encoder: {e}")
                raise RuntimeError(f"Failed to load label encoder: {e}")
        else:
            logger.error(f"Label encoder not found in any expected location")
            raise FileNotFoundError(f"Label encoder not found. Searched in: {', '.join(potential_paths)}")
    
    def classify(self, text, return_confidence=False, confidence_threshold=0.3):
        """
        Classify a text summary.
        
        Args:
            text: Text to classify
            return_confidence: Whether to return confidence scores
            confidence_threshold: Minimum confidence needed to make a prediction
            
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
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Set model to eval mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                # Handle different model output formats
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Check if outputs is a tuple/list or has a logits attribute
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    # Assume outputs are the logits directly
                    logits = outputs
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Create dictionary of category -> confidence
                confidence = {}
                for idx, prob in enumerate(probabilities):
                    if idx in self.idx_to_category:
                        category = self.idx_to_category[idx]
                        confidence[category] = prob.item()
                
                # Get predicted class with highest confidence 
                predicted_idx = torch.argmax(logits, dim=1).item()
                predicted_category = self.idx_to_category[predicted_idx]
                predicted_confidence = probabilities[predicted_idx].item()
                
                # If the highest confidence is below threshold, check if there's a
                # significant difference between top categories
                if predicted_confidence < confidence_threshold:
                    # Sort confidences to find top 2
                    sorted_confidence = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
                    
                    # If it's very close between top categories, consider it ambiguous
                    if len(sorted_confidence) >= 2:
                        top_diff = sorted_confidence[0][1] - sorted_confidence[1][1]
                        if top_diff < 0.05:  # Difference less than 5%
                            logger.info(f"Low confidence prediction ({predicted_confidence:.4f}) with small margin ({top_diff:.4f})")
                
                if return_confidence:
                    return predicted_category, confidence
                
                return predicted_category
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

def classify_summary(text, model_dir='./models/final'):
    """
    Classify a summary.
    
    Args:
        text: Text to classify
        model_dir: Directory containing the model
        
    Returns:
        Predicted category
    """
    try:
        classifier = SummaryClassifier(model_dir)
        return classifier.classify(text)
    except Exception as e:
        logger.error(f"Error in classify_summary: {e}")
        raise

def classify_batch(texts, model_dir='./models/final'):
    """
    Classify a batch of summaries.
    
    Args:
        texts: List of texts to classify
        model_dir: Directory containing the model
        
    Returns:
        List of predicted categories
    """
    try:
        # Input validation
        if not isinstance(texts, list):
            raise ValueError("Texts must be a list")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")
        
        # Initialize classifier once to avoid loading the model multiple times
        classifier = SummaryClassifier(model_dir)
        
        # Classify each text
        return [classifier.classify(text) for text in texts]
    except Exception as e:
        logger.error(f"Error in classify_batch: {e}")
        raise

def classify_with_confidence(text, model_dir='./models/final'):
    """
    Classify a summary and return confidence scores.
    
    Args:
        text: Text to classify
        model_dir: Directory containing the model
        
    Returns:
        Tuple of (predicted category, confidence scores)
    """
    try:
        classifier = SummaryClassifier(model_dir)
        return classifier.classify(text, return_confidence=True)
    except Exception as e:
        logger.error(f"Error in classify_with_confidence: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify a meeting summary. Only the most likely category will be shown.")
    parser.add_argument('text', type=str, help='Text to classify')
    parser.add_argument('--model_dir', type=str, default='./models/final',
                       help='Directory containing the model')
    args = parser.parse_args()

    try:
        category = classify_summary(args.text, args.model_dir)
        print(f"Category: {category}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)