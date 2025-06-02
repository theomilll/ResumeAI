import os
import sys

import joblib
from dotenv import load_dotenv  # For loading .env file for local development
from flask import Flask, jsonify, render_template, request

# Load environment variables from .env file if it exists
load_dotenv()

# Ensure src directory is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__))) # Ensure 'categorizacao' is on path for notion_module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ResumeAI_NotionExporter')))

from notion_exporter import NotionExporter
from src.predict import SummaryClassifier  # BERT model
# Import prediction modules and NotionExporter
from src.predict_tfidf import load_label_encoder as load_tfidf_encoder

app = Flask(__name__)

# --- Configuration ---
# Try to get Notion configuration from environment variables
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
# You can choose to export to a parent page or a database.
# Prioritize Database ID if both are set.
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")
NOTION_PARENT_PAGE_ID = os.environ.get("NOTION_PARENT_PAGE_ID")

if not NOTION_TOKEN:
    print("WARNING: NOTION_TOKEN environment variable not set. Notion export will fail.")
if not NOTION_DATABASE_ID and not NOTION_PARENT_PAGE_ID:
    print("WARNING: Neither NOTION_DATABASE_ID nor NOTION_PARENT_PAGE_ID environment variable is set. Notion export will fail.")

# --- Load Models ---
# TF-IDF Model (recommended for this integrated app due to speed and simplicity)
TFIDF_MODEL_PATH = os.path.join('models', 'tfidf', 'tfidf_pipeline.joblib')
TFIDF_ENCODER_PATH = os.path.join('models', 'tfidf', 'label_encoder.csv')

tfidf_pipeline = None
tfidf_encoder = None
try:
    tfidf_pipeline = joblib.load(TFIDF_MODEL_PATH)
    tfidf_encoder = load_tfidf_encoder(TFIDF_ENCODER_PATH)
    print("TF-IDF model and encoder loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: TF-IDF model or encoder not found at {TFIDF_MODEL_PATH} or {TFIDF_ENCODER_PATH}. Categorization will fail.")
except Exception as e:
    print(f"ERROR: Could not load TF-IDF model/encoder: {e}. Categorization will fail.")

# BERT Model (optional)
BERT_MODEL_PATH = os.path.join('models', 'final')
bert_classifier = None
try:
    bert_classifier = SummaryClassifier(BERT_MODEL_PATH)
    print("BERT classifier loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load BERT classifier: {e}. BERT predictions will not be available.")


@app.route('/', methods=['GET'])
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_text():
    """
    Handles text input (typed or transcribed), categorizes it, 
    and sends it to Notion.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    text_input = data.get('text_input', '').strip()
    model_type = data.get('model', 'tfidf')
    source_type = data.get('source_type')
    source_name = data.get('source_name')

    if not text_input:
        return jsonify({"status": "error", "message": "No text provided"}), 400

    # --- 1. Categorization ---
    category_result = "Unknown"
    if model_type == 'tfidf':
        if tfidf_pipeline and tfidf_encoder:
            try:
                predicted_index = tfidf_pipeline.predict([text_input])[0]
                category_result = tfidf_encoder.get(int(predicted_index), 'Unknown category index')
            except Exception as e:
                print(f"Error during TF-IDF prediction: {e}")
                return jsonify({"status": "error", "message": f"Error during TF-IDF categorization: {e}", "category": "Error"}), 500
        else:
            return jsonify({"status": "error", "message": "TF-IDF model not loaded.", "category": "Error"}), 500
    elif model_type == 'bert':
        if bert_classifier:
            try:
                category_result = bert_classifier.classify(text_input)
            except Exception as e:
                print(f"Error during BERT prediction: {e}")
                return jsonify({"status": "error", "message": f"Error during BERT classification: {e}", "category": "Error"}), 500
        else:
            return jsonify({"status": "error", "message": "BERT classifier not available.", "category": "Error"}), 500
    else:
        return jsonify({"status": "error", "message": f"Unknown model type: {model_type}", "category": "Error"}), 400

    # --- 2. Notion Export ---
    notion_url = None
    notion_message = "Notion export skipped (not configured or error)."

    if NOTION_TOKEN and (NOTION_DATABASE_ID or NOTION_PARENT_PAGE_ID):
        try:
            exporter = NotionExporter(token=NOTION_TOKEN)
            title = f"ResumeAI: {source_name if source_name else 'Nova Nota'}"
            
            # Prefer exporting to a database if ID is provided
            if NOTION_DATABASE_ID:
                response = exporter.create_database_item(
                    database_id=NOTION_DATABASE_ID,
                    title=title,
                    content=text_input,
                    categories=[category_result] if category_result != "Unknown" else [],
                    source_type=source_type,
                    source_name=source_name
                )
                notion_message = "Successfully exported to Notion Database."
            elif NOTION_PARENT_PAGE_ID: # Fallback to creating a page
                response = exporter.create_page(
                    parent_page_id=NOTION_PARENT_PAGE_ID,
                    title=title,
                    content=text_input,
                    categories=[category_result] if category_result != "Unknown" else [],
                    source_type=source_type,
                    source_name=source_name
                )
                notion_message = "Successfully exported to Notion Page."
            
            notion_url = response.get('url')
        except Exception as e:
            print(f"Error during Notion export: {e}")
            notion_message = f"Error during Notion export: {e}"
            return jsonify({
                "status": "error", 
                "message": notion_message,
                "category": category_result,
                "text": text_input
            }), 500
    else:
        print("Notion token or target ID not configured. Skipping export.")


    return jsonify({
        "status": "success",
        "message": notion_message,
        "category": category_result,
        "notion_url": notion_url,
        "text_processed": text_input
    })

if __name__ == '__main__':
    # For local development, you can create a .env file in the 'categorizacao' directory
    # with your NOTION_TOKEN and NOTION_PARENT_PAGE_ID or NOTION_DATABASE_ID
    # Example .env file:
    # NOTION_TOKEN="secret_yournotiontoken"
    # NOTION_PARENT_PAGE_ID="your_page_id_here" 
    # OR
    # NOTION_DATABASE_ID="your_database_id_here"

    app.run(debug=True, port=5000) # Runs on port 5000 by default
