import os
import json
import secrets
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename

from app.models.rag_model import RAGModel
from app.config import APP_PORT, SECRET_KEY, DATA_DIR

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = DATA_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
CORS(app)

# Initialize RAG model (this might take a moment as it loads the vector store)
rag_model = RAGModel()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'json', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Process a query and return a response"""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({
                'error': 'No query provided'
            }), 400
        
        # Process the query through the RAG model
        result = rag_model.process_query(user_query)
        
        return jsonify({
            'response': result['response'],
            'sources': result['sources'],
            'is_jailbreak': result['is_jailbreak'],
            'is_out_of_domain': result['is_out_of_domain']
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a new data file"""
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file part in the request'
            }), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        # Check if the file has an allowed extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the file and add it to the vector store
        result = rag_model.add_new_data(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/add-document', methods=['POST'])
def add_document():
    """Add a new document manually"""
    try:
        data = request.json
        
        # Validate required fields
        if 'category' not in data or 'question' not in data or 'answer' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: category, question, and answer are required'
            }), 400
        
        # Prepare document
        document = {
            'category': data['category'],
            'question': data['question'],
            'answer': data['answer'],
            'content': f"Question: {data['question']}\nAnswer: {data['answer']}",
            'source': 'manual_addition'
        }
        
        # Add document to vector store
        rag_model.add_document(document)
        
        return jsonify({
            'success': True,
            'message': 'Document added successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error adding document: {str(e)}'
        }), 500

def main():
    """Main entry point for the application"""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=APP_PORT, debug=True)

if __name__ == '__main__':
    main() 