import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional

class DataProcessor:
    """
    Process and sanitize bank customer service data from various sources.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the DataProcessor.
        
        Args:
            data_dir: Directory where data files are stored
        """
        self.data_dir = data_dir
        
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of documents extracted from the JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Process different JSON structures based on what's found
            if 'categories' in data:
                for category in data['categories']:
                    category_name = category.get('category', 'Uncategorized')
                    
                    for item in category.get('questions', []):
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        
                        if question and answer:
                            documents.append({
                                'category': category_name,
                                'question': question,
                                'answer': answer,
                                'content': f"Question: {question}\nAnswer: {answer}",
                                'source': os.path.basename(file_path)
                            })
            
            return documents
        
        except Exception as e:
            print(f"Error loading JSON data from {file_path}: {e}")
            return []
    
    def load_excel_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process data from Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of documents extracted from the Excel file
        """
        try:
            # Read all sheets in the Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            documents = []
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                # Clean column names (remove whitespace, lowercase)
                df.columns = [col.strip().lower() for col in df.columns]
                
                # Determine the structure by checking columns
                if 'question' in df.columns and 'answer' in df.columns:
                    # QA format
                    for _, row in df.iterrows():
                        question = row.get('question', '')
                        answer = row.get('answer', '')
                        category = row.get('category', sheet_name)
                        
                        if pd.notna(question) and pd.notna(answer):
                            documents.append({
                                'category': category if pd.notna(category) else sheet_name,
                                'question': question,
                                'answer': answer,
                                'content': f"Question: {question}\nAnswer: {answer}",
                                'source': os.path.basename(file_path)
                            })
                
                elif 'product' in df.columns and 'description' in df.columns:
                    # Product information format
                    for _, row in df.iterrows():
                        product_name = row.get('product', '')
                        description = row.get('description', '')
                        features = row.get('features', '')
                        
                        if pd.notna(product_name) and (pd.notna(description) or pd.notna(features)):
                            content = f"Product: {product_name}\n"
                            if pd.notna(description):
                                content += f"Description: {description}\n"
                            if pd.notna(features):
                                content += f"Features: {features}"
                            
                            documents.append({
                                'category': sheet_name,
                                'product': product_name,
                                'description': description if pd.notna(description) else '',
                                'features': features if pd.notna(features) else '',
                                'content': content,
                                'source': os.path.basename(file_path)
                            })
                
                else:
                    # If structure unknown, try to generalize
                    for _, row in df.iterrows():
                        content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                        
                        if content:
                            documents.append({
                                'category': sheet_name,
                                'content': content,
                                'source': os.path.basename(file_path)
                            })
            
            return documents
        
        except Exception as e:
            print(f"Error loading Excel data from {file_path}: {e}")
            return []
    
    def process_all_data(self) -> List[Dict[str, Any]]:
        """
        Process all data files in the data directory.
        
        Returns:
            List of all documents extracted from data files
        """
        all_documents = []
        
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            
            if os.path.isfile(file_path):
                if filename.lower().endswith('.json'):
                    documents = self.load_json_data(file_path)
                    all_documents.extend(documents)
                
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    documents = self.load_excel_data(file_path)
                    all_documents.extend(documents)
        
        return all_documents
    
    def anonymize_data(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize sensitive information in documents.
        This function can be extended to handle real anonymization requirements.
        
        Args:
            documents: List of documents to anonymize
            
        Returns:
            List of anonymized documents
        """
        # Implement actual anonymization logic here if needed
        # This implementation assumes data is already anonymized
        return documents
    
    def save_processed_data(self, documents: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save processed documents to a JSON file.
        
        Args:
            documents: List of documents to save
            output_file: Path to save the processed data
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            print(f"Processed data saved to {output_file}")
        
        except Exception as e:
            print(f"Error saving processed data to {output_file}: {e}")
    
    def load_processed_data(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load previously processed documents from a JSON file.
        
        Args:
            input_file: Path to the processed data file
            
        Returns:
            List of documents
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            return documents
        
        except Exception as e:
            print(f"Error loading processed data from {input_file}: {e}")
            return [] 