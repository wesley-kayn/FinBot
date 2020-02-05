from sentence_transformers import SentenceTransformer
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingProvider:
    """
    Provides embedding functionality with SentenceTransformer BGE-M3.
    """
    
    def __init__(self):
        """
        Initialize the embedding provider with BGE-M3.
        """
        # Get HuggingFace token from environment
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Initialize the BGE-M3 model
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", token=self.hf_token)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    # Add compatibility methods for LangChain (if you're using it)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (LangChain compatibility).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.get_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query (LangChain compatibility).
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.get_embedding(text)