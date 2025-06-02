# ResumeAI Integrated Application

A single-page application that integrates three key modules:
- **Speech-to-text transcription** using the browser's Web Speech API
- **Text categorization** using machine learning models (TF-IDF and BERT)
- **Notion export** for saving categorized content to Notion pages or databases

## Features

- Direct integration with existing modules from the `categorization` project
- Model selection between TF-IDF (fast) and BERT (more accurate)
- Speech recording and transcription using browser APIs
- Categorization of text content with confidence scores
- Export to Notion pages or databases with proper metadata

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables (create a `.env` file):
   ```
   NOTION_TOKEN=your_notion_integration_token
   NOTION_DATABASE_ID=your_notion_database_id
   NOTION_PARENT_PAGE_ID=your_notion_parent_page_id
   FLASK_ENV=development  # Use 'production' for production
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

## Architecture

This application directly integrates the existing modules:

- **Categorization**: Uses both TF-IDF and BERT models from `categorization/categorizacao`
- **Notion Export**: Uses the `NotionExporter` class from `categorization/ResumeAI_NotionExporter`
- **Speech-to-Text**: Integrates browser Web Speech API with minimal backend support

## Usage

1. Choose between text input or speech recording
2. Select the preferred categorization model (TF-IDF or BERT)
3. Process the content to get categorization results
4. Export to Notion if desired

## Requirements

- Python 3.8+
- Flask
- PyTorch with CUDA support (optional, for better BERT performance)
- Modern web browser with Speech Recognition API support (Chrome recommended)
