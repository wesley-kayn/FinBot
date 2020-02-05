import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from langchain.schema.document import Document
from app.utils.embeddings import EmbeddingProvider
from pathlib import Path

class VectorStore:
    """
    Manages the vector store for document retrieval using FAISS.
    """
    
    def __init__(self, vector_store_path="app/data/vector_store"):
        """
        Initialize the vector store with the embedding provider.
        """
        self.embedding_provider = EmbeddingProvider()
        self.vector_store_path = vector_store_path
        self.index_path = os.path.join(vector_store_path, "faiss.index")
        self.meta_path = os.path.join(vector_store_path, "meta.pkl")
        self.index = None
        self.metadata = []
    
    def _convert_to_documents(self, items: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert dictionary items to Document objects for the vector store.
        
        Args:
            items: List of dictionary items to convert
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for item in items:
            # Extract the text content
            content = item.get('content', '')
            
            # Create metadata from the rest of the fields
            metadata = {k: v for k, v in item.items() if k != 'content'}
            
            # Create a Document
            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            documents.append(document)
        
        return documents
    
    def create_vector_store(self, items: List[Dict[str, Any]]) -> None:
        """
        Create a new vector store from the provided items.
        
        Args:
            items: List of dictionary items to add to the vector store
        """
        # Extract content for embedding
        texts = [item.get('content', '') for item in items]
        
        # Add metadata for each item (excluding content)
        metadata = []
        for item in items:
            meta = {k: v for k, v in item.items() if k != 'content'}
            metadata.append(meta)
        
        # Get embeddings
        embeddings = self.embedding_provider.get_embeddings(texts)
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_np)
        
        # Store content and metadata
        self.metadata = []
        for i, text in enumerate(texts):
            self.metadata.append({
                'content': text,
                **metadata[i]
            })
        
        # Save to disk
        self.save_vector_store()
        
        print(f"Vector store created with {len(texts)} items")
    
    def add_to_vector_store(self, items: List[Dict[str, Any]]) -> None:
        """
        Add items to an existing vector store.
        
        Args:
            items: List of dictionary items to add to the vector store
        """
        if self.index is None:
            # If no index exists, create a new one
            return self.create_vector_store(items)
        
        # Extract content for embedding
        texts = [item.get('content', '') for item in items]
        
        # Get embeddings
        embeddings = self.embedding_provider.get_embeddings(texts)
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        
        # Add to metadata
        for i, item in enumerate(items):
            meta = {k: v for k, v in item.items() if k != 'content'}
            self.metadata.append({
                'content': texts[i],
                **meta
            })
        
        # Save the updated vector store
        self.save_vector_store()
    
    def save_vector_store(self) -> None:
        """
        Save the vector store to disk.
        """
        if self.index is None:
            print("No vector store to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Vector store saved to {self.vector_store_path}")
    
    def load_vector_store(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if the vector store was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                print(f"Vector store loaded with {len(self.metadata)} items")
                return True
            else:
                print(f"Vector store files not found at {self.vector_store_path}")
                return False
        
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search against the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if self.index is None:
            print("No vector store loaded. Please create or load a vector store first.")
            return []
        
        # Get query embedding
        query_embedding = self.embedding_provider.get_embedding(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search the index
        scores, indices = self.index.search(query_embedding_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            # Get metadata and content for this result
            item = self.metadata[idx]
            content = item.get('content', '')
            meta = {k: v for k, v in item.items() if k != 'content'}
            
            # Create a Document
            document = Document(
                page_content=content,
                metadata={
                    **meta,
                    'score': float(scores[0][i])
                }
            )
            
            results.append(document)
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform a similarity search against the vector store and return scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        documents = self.similarity_search(query, k=k)
        return [(doc, doc.metadata.get('score', 0.0)) for doc in documents]