from typing import Dict, List, Any, Optional
from app.utils.vector_store import VectorStore
from app.utils.llm_provider import LLMProvider
from app.utils.data_processor import DataProcessor
from app.utils.excel_parser import build_index_from_excel
from app.config import DATA_DIR
import os
import time
import json
from app.utils.metrics import update_session_metrics

class RAGModel:
    """
    Retrieval-Augmented Generation model that combines vector retrieval and LLM generation.
    """
    
    def __init__(self):
        """
        Initialize the RAG model with required components.
        """
        self.vector_store = VectorStore()
        self.llm_provider = LLMProvider()
        self.rag_chain = self.llm_provider.create_rag_chain()
        self.load_or_create_vector_store()
    
    def load_or_create_vector_store(self) -> None:
        """
        Load an existing vector store or create a new one if none exists.
        """
        # Try to load the vector store
        loaded = self.vector_store.load_vector_store()
        
        # Check for the core bank knowledge file and process it with the new method
        bank_knowledge_path = os.path.join(DATA_DIR, "NUST Bank-Product-Knowledge.xlsx")
        if os.path.exists(bank_knowledge_path):
            print(f"Ensuring core bank knowledge file is processed...")
            build_index_from_excel(bank_knowledge_path)
            # Reload the vector store after processing
            self.vector_store.load_vector_store()
            return
            
        # If loading fails, create a new vector store from data
        if not loaded:
            print("Creating new vector store...")
            data_processor = DataProcessor(DATA_DIR)
            documents = data_processor.process_all_data()
            
            if documents:
                self.vector_store.create_vector_store(documents)
                print(f"Created vector store with {len(documents)} documents")
            else:
                print("No documents found to create vector store")
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """
        Add a new document to the vector store.
        
        Args:
            document: Dictionary containing document information
        """
        # Add the document to the vector store
        self.vector_store.add_to_vector_store([document])
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing response and related information
        """
        start_time = time.time()
        
        # Initialize metrics dictionary
        metrics = {
            "query_length": len(query),
            "retrieved_docs": 0,
            "similarity_scores": [],
            "retrieval_time_ms": 0,
            "generation_time_ms": 0,
            "total_time_ms": 0
        }
        
        # Check for jailbreak attempts
        if self.llm_provider.handle_jailbreaking(query):
            response = (
                "I cannot process this request as it appears to be attempting to bypass "
                "my operational guidelines. Please submit a valid banking inquiry."
            )
            total_time = time.time() - start_time
            metrics["total_time_ms"] = round(total_time * 1000, 2)
            print(f"[METRICS] {json.dumps(metrics)}")
            return {
                "response": response,
                "sources": [],
                "is_jailbreak": True,
                "is_out_of_domain": False,
                "metrics": metrics
            }
        
        # Check if query is out of domain
        if self.llm_provider.is_out_of_domain(query):
            response = self.llm_provider.handle_out_of_domain_query(query)
            total_time = time.time() - start_time
            metrics["total_time_ms"] = round(total_time * 1000, 2)
            print(f"[METRICS] {json.dumps(metrics)}")
            return {
                "response": response,
                "sources": [],
                "is_jailbreak": False,
                "is_out_of_domain": True,
                "metrics": metrics
            }
        
        # Get relevant documents from vector store
        retrieval_start = time.time()
        docs = self.vector_store.similarity_search(query, k=3)
        retrieval_time = time.time() - retrieval_start
        
        # Track sources for attribution
        sources = []
        for doc in docs:
            if 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
        
        # If no relevant documents found
        if not docs:
            response = (
                "I don't have enough information to answer this question. "
                "Please contact our customer service at +92 (51) 111 000 494 for assistance."
            )
            # Calculate total time even when no docs found
            total_time = time.time() - start_time
            metrics["total_time_ms"] = round(total_time * 1000, 2)
            print(f"[METRICS] {json.dumps(metrics)}")
        else:
            # Format documents for the prompt
            context = self.llm_provider.format_docs(docs)
            
            # Generate response using RAG chain
            generation_start = time.time()
            response = self.rag_chain.invoke({
                "context": context,
                "question": query
            })
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Log metrics to terminal
            metrics["retrieved_docs"] = len(docs)
            metrics["similarity_scores"] = [doc.metadata.get('score', 0) for doc in docs]
            metrics["retrieval_time_ms"] = round(retrieval_time * 1000, 2)
            metrics["generation_time_ms"] = round(generation_time * 1000, 2)
            metrics["total_time_ms"] = round(total_time * 1000, 2)
            
            print(f"[METRICS] {json.dumps(metrics)}")
        
        return {
            "response": response,
            "sources": sources,
            "is_jailbreak": False,
            "is_out_of_domain": False,
            "metrics": metrics
        }
    
    def add_new_data(self, filepath: str) -> Dict[str, Any]:
        """
        Process and add new data from a file.
        
        Args:
            filepath: Path to the file to process
            
        Returns:
            Dictionary with status information
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return {
                    "success": False,
                    "message": f"File not found: {filepath}"
                }
            
            # Process Excel files with the new parser
            if filepath.lower().endswith(('.xlsx', '.xls')):
                result = build_index_from_excel(filepath)
                
                if result:
                    # Need to reload the vector store after building
                    self.vector_store.load_vector_store()
                    
                    return {
                        "success": True,
                        "message": f"Successfully processed Excel file: {os.path.basename(filepath)}"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to process Excel file: {os.path.basename(filepath)}"
                    }
            
            # Process other file types with existing methods
            elif filepath.lower().endswith('.json'):
                # Create data processor instance
                data_processor = DataProcessor("")
                documents = data_processor.load_json_data(filepath)
                
                # Check if any documents were extracted
                if not documents:
                    return {
                        "success": False,
                        "message": f"No valid data found in file: {filepath}"
                    }
                
                # Add documents to vector store
                self.vector_store.add_to_vector_store(documents)
                
                return {
                    "success": True,
                    "message": f"Successfully added {len(documents)} documents from {filepath}",
                    "document_count": len(documents)
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unsupported file format: {filepath}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing file: {str(e)}"
            }
