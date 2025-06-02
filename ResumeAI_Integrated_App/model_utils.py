"""
Helper functions for loading models in a backward-compatible way.
"""
import os
import csv
import logging
import pickle
import joblib
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class TfidfBackupClassifier:
    """
    A simple TF-IDF + LogisticRegression classifier that can be used as a backup 
    if we can't load the original model.
    """
    def __init__(self, categories=None):
        self.categories = categories or {
            0: "Achados de Pesquisa",
            1: "Atualizações de Projeto",
            2: "Gestão de Equipe", 
            3: "Outras",
            4: "Reuniões com Clientes"
        }
        self.vectorizer = TfidfVectorizer(
            max_features=10000, 
            min_df=2, 
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(
            multi_class='ovr',
            C=1.0,
            solver='liblinear'
        )
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        self._fitted = False
        self.fallback_category_id = 3  # 'Outras' category
        
    def fit(self, X, y):
        """Fit the pipeline with training data (not used in our case)"""
        self.pipeline.fit(X, y)
        self._fitted = True
        return self
        
    def predict(self, X):
        """
        Predict category for text input.
        If not fitted, returns fallback category.
        """
        if not self._fitted:
            logger.warning("Using fallback prediction as model is not fitted")
            if isinstance(X, str):
                return np.array([self.fallback_category_id])
            return np.array([self.fallback_category_id] * len(X))
            
        try:
            return self.pipeline.predict(X)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            if isinstance(X, str):
                return np.array([self.fallback_category_id])
            return np.array([self.fallback_category_id] * len(X))

def load_safe_pipeline(file_path):
    """
    Try to load a pipeline model safely handling version compatibility issues.
    """
    try:
        # Try direct loading first
        return joblib.load(file_path)
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Standard loading failed: {e}. Trying alternative approach...")
        
        # Attempt a more forgiving load approach
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded with alternative method: {type(model)}")
                return model
        except Exception as e2:
            logger.error(f"Alternative loading also failed: {e2}")
            return None

def parse_csv_label_encoder(file_path):
    """
    Parse a CSV file containing label encoder mappings.
    Format expected: category,idx (column order varies)
    Returns a dictionary mapping indices to labels.
    """
    if not os.path.exists(file_path):
        logger.error(f"Label encoder file not found: {file_path}")
        return {}
        
    encoder_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.info(f"Reading label encoder from {file_path}")
            content = f.read()
            logger.info(f"CSV content preview: {content[:200]}...")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)  # Get headers
            
            # Determine column positions based on headers
            if headers and len(headers) >= 2:
                logger.info(f"CSV headers: {headers}")
                if 'category' in headers and 'idx' in headers:
                    cat_idx = headers.index('category')
                    num_idx = headers.index('idx')
                    logger.info(f"Format detected: category at position {cat_idx}, index at position {num_idx}")
                else:
                    # Default positions
                    cat_idx = 1
                    num_idx = 0
                    logger.info("Using default column positions: index=0, category=1")
            else:
                # Default positions
                cat_idx = 1
                num_idx = 0
                logger.info("No headers found, using default column positions")
            
            # Parse rows
            for row in reader:
                if len(row) >= 2:
                    try:
                        category = row[cat_idx]
                        index = int(row[num_idx])
                        encoder_dict[index] = category
                        logger.info(f"Parsed mapping: {index} -> {category}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing row {row}: {e}")
                        # Try the reverse order as fallback
                        try:
                            category = row[1 - cat_idx]
                            index = int(row[1 - num_idx])
                            encoder_dict[index] = category
                            logger.info(f"Fallback parsed mapping: {index} -> {category}")
                        except (ValueError, IndexError):
                            logger.warning(f"Fallback parsing also failed for row: {row}")
                            
        logger.info(f"Loaded {len(encoder_dict)} labels from {file_path}")
        
        if not encoder_dict:
            # Provide default categories if nothing could be loaded
            logger.warning("Using default categories as fallback")
            encoder_dict = {
                0: "Achados de Pesquisa",
                1: "Atualizações de Projeto",
                2: "Gestão de Equipe", 
                3: "Outras",
                4: "Reuniões com Clientes"
            }
        
        return encoder_dict
    except Exception as e:
        logger.error(f"Error loading label encoder: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide default categories as fallback
        logger.warning("Using default categories due to parsing error")
        return {
            0: "Achados de Pesquisa",
            1: "Atualizações de Projeto",
            2: "Gestão de Equipe", 
            3: "Outras",
            4: "Reuniões com Clientes"
        }
