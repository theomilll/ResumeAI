import os
import sys
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import joblib

# Import our custom model utils
from model_utils import load_safe_pipeline, parse_csv_label_encoder, TfidfBackupClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add the necessary paths for importing existing modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
CATEGORIZATION_PATH = os.path.join(PARENT_DIR, 'categorizacao')
NOTION_PATH = os.path.join(PARENT_DIR, 'ResumeAI_NotionExporter')
STT_PATH = os.path.join(PARENT_DIR, 'speech-to-text')

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Parent directory: {PARENT_DIR}")
logger.info(f"Categorization path: {CATEGORIZATION_PATH}")
logger.info(f"Notion path: {NOTION_PATH}")
logger.info(f"STT path: {STT_PATH}")

# Add paths to system path for importing
sys.path.append(CATEGORIZATION_PATH)
sys.path.append(os.path.join(CATEGORIZATION_PATH, 'src'))
sys.path.append(NOTION_PATH)
sys.path.append(STT_PATH)

# First clear any duplicate paths to avoid conflicts
sys_paths = set(sys.path)
sys.path = list(sys_paths)

# Add paths to system path for importing - adding each only once
for path in [CATEGORIZATION_PATH, os.path.join(CATEGORIZATION_PATH, 'src'), NOTION_PATH, STT_PATH]:
    if path not in sys.path:
        sys.path.append(path)
        logger.info(f"Added to sys.path: {path}")

# Import with proper error handling
try:
    # First import the Notion exporter - direct import from the module file
    sys.path.insert(0, NOTION_PATH)
    from notion_exporter import NotionExporter
    logger.info("Successfully imported NotionExporter")
except ImportError as e:
    logger.error(f"Failed to import NotionExporter: {e}")
    # Fallback direct import approach
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("notion_exporter", os.path.join(NOTION_PATH, "notion_exporter.py"))
        notion_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notion_module)
        NotionExporter = notion_module.NotionExporter
        logger.info("Successfully imported NotionExporter using spec loader")
    except Exception as e:
        logger.error(f"Failed to import NotionExporter using spec loader: {e}")
        NotionExporter = None

# Import the categorization modules
try:
    sys.path.insert(0, os.path.join(CATEGORIZATION_PATH, 'src'))
    from predict import SummaryClassifier
    from predict_tfidf import load_label_encoder as load_tfidf_encoder
    logger.info("Successfully imported prediction modules")
except ImportError as e:
    logger.error(f"Failed to import prediction modules: {e}")
    SummaryClassifier = None
    load_tfidf_encoder = None

app = Flask(__name__)

# --- Configuration ---
DEBUG = os.environ.get('FLASK_ENV') == 'development'
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
NOTION_PARENT_PAGE_ID = os.environ.get("NOTION_PARENT_PAGE_ID", "")

# --- Model Paths ---
TFIDF_MODEL_PATH = os.path.join(CATEGORIZATION_PATH, 'models', 'tfidf', 'tfidf_pipeline.joblib')
TFIDF_ENCODER_PATH = os.path.join(CATEGORIZATION_PATH, 'models', 'tfidf', 'label_encoder.csv')
BERT_MODEL_PATH = os.path.join(CATEGORIZATION_PATH, 'models', 'final')

logger.info(f"TF-IDF model path: {TFIDF_MODEL_PATH}")
logger.info(f"TF-IDF encoder path: {TFIDF_ENCODER_PATH}")
logger.info(f"BERT model path: {BERT_MODEL_PATH}")

# Check if model files exist
for path in [TFIDF_MODEL_PATH, TFIDF_ENCODER_PATH, BERT_MODEL_PATH]:
    if os.path.exists(path):
        logger.info(f"Found model file: {path}")
    else:
        logger.warning(f"Model file does not exist: {path}")

# --- Load Models ---
# Load TF-IDF model
tfidf_pipeline = None
tfidf_encoder = None
try:
    if os.path.exists(TFIDF_MODEL_PATH):
        # Use our safe loader instead of joblib directly
        logger.info("Loading TF-IDF model with safe loader...")
        tfidf_pipeline = load_safe_pipeline(TFIDF_MODEL_PATH)
        
        # Use our own CSV parser instead of the module function
        tfidf_encoder = parse_csv_label_encoder(TFIDF_ENCODER_PATH)
        
        if tfidf_pipeline and tfidf_encoder:
            logger.info("TF-IDF model and encoder loaded successfully.")
            # Verify encoder format
            if tfidf_encoder is not None:
                logger.info(f"TF-IDF encoder type: {type(tfidf_encoder)}")
                if isinstance(tfidf_encoder, dict) and tfidf_encoder:
                    logger.info(f"TF-IDF encoder first few entries: {list(tfidf_encoder.items())[:3]}")
        else:
            if not tfidf_pipeline:
                logger.warning("Failed to load original TF-IDF pipeline, using backup classifier")
                tfidf_pipeline = TfidfBackupClassifier(tfidf_encoder)
                logger.info("Created backup TF-IDF classifier")
            if not tfidf_encoder:
                logger.warning("Failed to load TF-IDF encoder, using default categories")
                if hasattr(tfidf_pipeline, 'categories'):
                    tfidf_encoder = tfidf_pipeline.categories
                    logger.info(f"Using backup classifier categories: {tfidf_encoder}")
    else:
        logger.warning(f"TF-IDF model path does not exist: {TFIDF_MODEL_PATH}")
        # Create backup classifier with default categories
        logger.info("Creating backup TF-IDF classifier with default categories")
        tfidf_pipeline = TfidfBackupClassifier()
        tfidf_encoder = tfidf_pipeline.categories
except Exception as e:
    logger.error(f"TF-IDF model loading failed: {e}")
    import traceback
    traceback.print_exc()
    # Create backup classifier as fallback
    logger.info("Creating backup TF-IDF classifier after exception")
    tfidf_pipeline = TfidfBackupClassifier()
    tfidf_encoder = tfidf_pipeline.categories

# Load BERT model
bert_classifier = None
try:
    if os.path.exists(BERT_MODEL_PATH) and SummaryClassifier:
        logger.info(f"Loading BERT model from {BERT_MODEL_PATH}...")
        bert_classifier = SummaryClassifier(BERT_MODEL_PATH)
        logger.info("BERT classifier loaded successfully.")
    else:
        logger.warning("BERT model path does not exist or class not available")
except Exception as e:
    logger.error(f"BERT model loading failed: {e}")
    import traceback
    traceback.print_exc()

# --- Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# --- API Routes ---

@app.route('/api/speech-config', methods=['GET'])
def get_speech_config():
    """Return speech recognition configuration for the frontend"""
    return jsonify({
        "language": "pt-BR",  # Default language
        "continuous": True,
        "interimResults": False,
        "maxAlternatives": 1
    })

@app.route('/api/list-models', methods=['GET'])
def list_models():
    """List available categorization models"""
    models = []
    
    if tfidf_pipeline and tfidf_encoder:
        models.append({
            "id": "tfidf", 
            "name": "TF-IDF Model", 
            "description": "Faster text classification model using TF-IDF",
            "available": True
        })
    else:
        models.append({
            "id": "tfidf", 
            "name": "TF-IDF Model (Unavailable)", 
            "description": "Model could not be loaded",
            "available": False
        })
    
    if bert_classifier:
        models.append({
            "id": "bert", 
            "name": "BERT Model", 
            "description": "More accurate deep learning model (slower)",
            "available": True
        })
    else:
        models.append({
            "id": "bert", 
            "name": "BERT Model (Unavailable)", 
            "description": "Model could not be loaded",
            "available": False
        })
    
    return jsonify({"models": models})

@app.route('/api/categorize', methods=['POST'])
def categorize_text():
    """Categorize text using the selected model"""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 400
    
    data = request.get_json()
    text = data.get('text', '').strip()
    model_type = data.get('model', 'tfidf')  # Default to tfidf if not specified
    include_confidence = data.get('include_confidence', False)
    
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400
    
    logger.info(f"Categorizing text using {model_type} model, length: {len(text)} chars")
    
    try:
        if model_type == 'tfidf':
            if not tfidf_pipeline or not tfidf_encoder:
                logger.error("TF-IDF model or encoder not loaded")
                return jsonify({"success": False, "error": "TF-IDF model not loaded"}), 500
            
            logger.info(f"Using TF-IDF pipeline type: {type(tfidf_pipeline).__name__}")
            
            # Get prediction
            try:
                predicted_index = int(tfidf_pipeline.predict([text])[0])
                logger.info(f"Raw prediction index: {predicted_index}")
            except Exception as e:
                logger.error(f"Error during TF-IDF prediction: {e}")
                # Use fallback category
                predicted_index = 3  # Usually 'Outras'
                logger.info("Using fallback prediction index after error")
            
            # Map to category name
            if isinstance(tfidf_encoder, dict):
                category = tfidf_encoder.get(predicted_index, 'Outras')
                logger.info(f"Mapped to category: {category}")
            else:
                category = 'Outras'
                logger.warning("Encoder not available, using default category")
            
            # Calculate confidence score if requested
            confidence = None
            if include_confidence:
                # Get probabilities if available
                if hasattr(tfidf_pipeline, 'predict_proba') and not isinstance(tfidf_pipeline, TfidfBackupClassifier):
                    try:
                        proba = tfidf_pipeline.predict_proba([text])[0]
                        confidence = float(proba.max())
                        logger.info(f"Confidence from predict_proba: {confidence}")
                    except Exception as e:
                        logger.error(f"Error getting probabilities: {e}")
                        confidence = 0.7
                else:
                    # Default confidence if not available
                    confidence = 0.8 if category != 'Outras' else 0.6
                    logger.info(f"Using default confidence: {confidence}")
            
            result = {"success": True, "category": category}
            if include_confidence:
                result["confidence"] = confidence
            logger.info(f"TF-IDF result: {result}")
            return jsonify(result)
            
        elif model_type == 'bert':
            if not bert_classifier:
                logger.error("BERT classifier not loaded")
                return jsonify({"success": False, "error": "BERT model not loaded"}), 500
            
            logger.info("Using BERT model for prediction")
            if include_confidence:
                # Use the method that returns confidence scores
                try:
                    result = bert_classifier.predict_with_confidence(text)
                    logger.info(f"BERT prediction with confidence: {result}")
                    return jsonify({
                        "success": True, 
                        "category": result['predicted_category'],
                        "confidence": result['confidence'],
                        "confidences": result.get('confidences', {})
                    })
                except Exception as e:
                    logger.error(f"BERT prediction with confidence error: {e}")
                    # Use our fallback in case of errors
                    return jsonify({
                        "success": True, 
                        "category": "Outras",
                        "confidence": 0.5,
                        "note": "Fallback prediction"
                    })
            else:
                try:
                    category = bert_classifier.classify(text)
                    logger.info(f"BERT classification result: {category}")
                    return jsonify({"success": True, "category": category})
                except Exception as e:
                    logger.error(f"BERT classification error: {e}")
                    return jsonify({"success": True, "category": "Outras", "note": "Fallback after error"})
        else:
            logger.warning(f"Unknown model type requested: {model_type}")
            return jsonify({"success": False, "error": f"Unknown model type: {model_type}"}), 400
    
    except Exception as e:
        logger.error(f"Global error during categorization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/export/notion', methods=['POST'])
def export_to_notion():
    """Export content to Notion"""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 400
    
    data = request.get_json()
    token = data.get('token', NOTION_TOKEN)
    destination_id = data.get('destination_id', '')
    export_type = data.get('export_type', 'page')  # 'page' or 'database'
    title = data.get('title', 'ResumeAI Export')
    content = data.get('content', '')
    categories = data.get('categories', [])
    source_type = data.get('source_type', 'text')
    source_name = data.get('source_name', '')
    language = data.get('language', 'pt-BR')
    
    if not token:
        return jsonify({"success": False, "error": "Notion token is required"}), 400
    
    if not destination_id:
        return jsonify({"success": False, "error": "Destination ID is required"}), 400
    
    if not content:
        return jsonify({"success": False, "error": "Content is required"}), 400
    
    try:
        notion = NotionExporter(token)
        
        if export_type == 'page':
            # Create a new page in the specified parent page
            page_id = notion.create_page(
                title=title,
                parent_id=destination_id,
                content=content,
                categories=categories,
                source_type=source_type,
                language=language
            )
            
            if page_id:
                # Try to construct a URL to the page
                notion_url = f"https://notion.so/{page_id.replace('-', '')}"
                return jsonify({
                    "success": True, 
                    "message": "Content exported to Notion page", 
                    "page_id": page_id,
                    "notion_url": notion_url
                })
            else:
                return jsonify({"success": False, "error": "Failed to create Notion page"}), 500
        
        elif export_type == 'database':
            # Add an item to a database
            item_id = notion.create_database_item(
                database_id=destination_id,
                title=title,
                content=content,
                categories=categories,
                source_type=source_type,
                source_name=source_name,
                language=language
            )
            
            if item_id:
                # Try to construct a URL to the database item
                notion_url = f"https://notion.so/{item_id.replace('-', '')}"
                return jsonify({
                    "success": True, 
                    "message": "Content exported to Notion database", 
                    "item_id": item_id,
                    "notion_url": notion_url
                })
            else:
                return jsonify({"success": False, "error": "Failed to add item to Notion database"}), 500
        
        else:
            return jsonify({"success": False, "error": f"Invalid export type: {export_type}"}), 400
    
    except Exception as e:
        print(f"Error exporting to Notion: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process():
    """
    Process input content:
    1. Categorize text using the selected model
    2. Optionally export to Notion
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 400
    
    data = request.get_json()
    text = data.get('text', '').strip()
    model_type = data.get('model', 'tfidf')
    export_to_notion = data.get('export_to_notion', False)
    
    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400
        
    try:
        # Step 1: Categorize the text
        category_result = None
        
        try:
            if model_type == 'tfidf':
                if not tfidf_pipeline or not tfidf_encoder:
                    return jsonify({"success": False, "error": "TF-IDF model not loaded"}), 500
                    
                logger.info(f"Predicting with TF-IDF model: {text[:50]}...")
                predicted_index = tfidf_pipeline.predict([text])[0]
                predicted_index_int = int(predicted_index)
                logger.info(f"TF-IDF prediction index: {predicted_index_int}")
                
                # Handle different encoder formats
                if isinstance(tfidf_encoder, dict):
                    category_result = tfidf_encoder.get(predicted_index_int, 'Unknown')
                else:
                    # If it's a different format, try to access it differently
                    try:
                        category_result = tfidf_encoder[predicted_index_int]
                    except (TypeError, KeyError, IndexError):
                        category_result = 'Unknown'
                        
                logger.info(f"TF-IDF predicted category: {category_result}")
                
            elif model_type == 'bert':
                if not bert_classifier:
                    return jsonify({"success": False, "error": "BERT model not loaded"}), 500
                
                logger.info(f"Predicting with BERT model: {text[:50]}...")
                try:
                    category_result = bert_classifier.classify(text)
                    logger.info(f"BERT predicted category: {category_result}")
                except Exception as bert_error:
                    logger.error(f"BERT classification error: {bert_error}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({"success": False, "error": f"BERT classification error: {str(bert_error)}"}), 500
        except Exception as e:
            logger.error(f"Error during categorization: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Error during categorization: {str(e)}"}), 500
            
        else:
            return jsonify({"success": False, "error": f"Unknown model type: {model_type}"}), 400
        
        result = {
            "success": True,
            "category": category_result,
            "text": text
        }
        
        # Step 2: Export to Notion if requested
        if export_to_notion:
            notion_token = data.get('notion_token', NOTION_TOKEN)
            destination_id = data.get('destination_id', NOTION_DATABASE_ID or NOTION_PARENT_PAGE_ID)
            export_type = data.get('export_type', 'page')
            title = data.get('title', f"ResumeAI: {category_result}")
            
            if not notion_token:
                result["notion_export"] = {"success": False, "error": "Notion token is required"}
                return jsonify(result)
                
            if not destination_id:
                result["notion_export"] = {"success": False, "error": "Destination ID is required"}
                return jsonify(result)
                
            try:
                notion = NotionExporter(notion_token)
                
                if export_type == 'page':
                    page_id = notion.create_page(
                        title=title,
                        parent_id=destination_id,
                        content=text,
                        categories=[category_result],
                        source_type=data.get('source_type', 'text'),
                        language=data.get('language', 'pt-BR')
                    )
                    
                    if page_id:
                        notion_url = f"https://notion.so/{page_id.replace('-', '')}"
                        result["notion_export"] = {
                            "success": True,
                            "page_id": page_id,
                            "notion_url": notion_url
                        }
                    else:
                        result["notion_export"] = {"success": False, "error": "Failed to create Notion page"}
                
                elif export_type == 'database':
                    item_id = notion.create_database_item(
                        database_id=destination_id,
                        title=title,
                        content=text,
                        categories=[category_result],
                        source_type=data.get('source_type', 'text'),
                        source_name=data.get('source_name', 'ResumeAI'),
                        language=data.get('language', 'pt-BR')
                    )
                    
                    if item_id:
                        notion_url = f"https://notion.so/{item_id.replace('-', '')}"
                        result["notion_export"] = {
                            "success": True,
                            "item_id": item_id,
                            "notion_url": notion_url
                        }
                    else:
                        result["notion_export"] = {"success": False, "error": "Failed to add item to Notion database"}
                
                else:
                    result["notion_export"] = {"success": False, "error": f"Invalid export type: {export_type}"}
            
            except Exception as e:
                print(f"Error exporting to Notion: {e}")
                result["notion_export"] = {"success": False, "error": str(e)}
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Use port 5001 to avoid conflicts with AirPlay
    print(f"ResumeAI Integrated App starting on port {port}...")
    print(f"Categorization module path: {CATEGORIZATION_PATH}")
    print(f"Notion module path: {NOTION_PATH}")
    print(f"Speech-to-Text module path: {STT_PATH}")
    app.run(host="0.0.0.0", port=port, debug=DEBUG)
