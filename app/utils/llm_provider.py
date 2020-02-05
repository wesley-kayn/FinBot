from typing import Dict, List, Any, Optional
import os
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import Runnable
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document
from dotenv import load_dotenv
from app.config import (
    LLM_SERVICE,
    OLLAMA_SERVER,
    HF_MODEL_ID  # Add this to your config.py
)

# Load environment variables
load_dotenv()

class LLMProvider:
    """
    Provides LLM functionality for generating responses using LLM services.
    Supports Ollama and HuggingFace API.
    """
    
    def __init__(self):
        """
        Initialize the LLM provider based on configuration.
        """
        # Debug print to verify config
        print(f"Initializing LLMProvider with service={LLM_SERVICE}, model={HF_MODEL_ID}")
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> Runnable:
        """
        Initialize the LLM based on configuration.
        
        Returns:
            An initialized LLM instance
        """
        if LLM_SERVICE.lower() == "ollama":
            print(f"Using Ollama with model: {HF_MODEL_ID}")
            return Ollama(
                model=HF_MODEL_ID,  # Use HF_MODEL_ID for both services
                base_url=OLLAMA_SERVER,
                temperature=0.1
            )
        elif LLM_SERVICE.lower() == "huggingface":
            # Get HuggingFace API token from environment variables
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is required for HuggingFace API")
            
            # Debug print the endpoint URL being used
            endpoint_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
            print(f"Using HuggingFace with model: {HF_MODEL_ID}")
            print(f"Endpoint URL: {endpoint_url}")
            
            # Pass parameters directly to the class instead of nesting them in model_kwargs
            return HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=hf_token,
                task="text-generation",
                temperature=0.1,
                max_new_tokens=512,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False
            )
        else:
            raise ValueError(f"Unsupported LLM service: {LLM_SERVICE}. Supported services: 'ollama', 'huggingface'")
    
    def create_rag_chain(self) -> Runnable:
        """
        Create a RAG (Retrieval-Augmented Generation) chain.
        
        Returns:
            A runnable RAG chain
        """
        # Define the prompt template for RAG
        prompt_template = """
        You are an AI assistant for Finbot, a trusted financial institution. Your role is to provide helpful, accurate, and professional responses to customer inquiries about the bank's products, services, and procedures.
        
        Use the following pieces of bank information to answer the question at the end.
        If you don't know the answer, just say "I don't have enough information to answer this question. Please contact our customer service at +92 (51) 111 000 494 for assistance." Don't try to make up an answer.
        
        {context}
        
        Question: {question}
        
        Helpful Answer:
        """
        
        # Create the prompt from the template
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the RAG chain
        rag_chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Format a list of documents into a string for use in prompts.
        
        Args:
            docs: List of Document objects
            
        Returns:
            Formatted string containing document contents
        """
        return "\n\n".join([doc.page_content for doc in docs])
    
    def apply_content_filtering(self, response: str) -> str:
        """
        Apply content filtering to ensure responses do not include sensitive information.
        This is a placeholder implementation that can be expanded with actual filtering logic.
        
        Args:
            response: The LLM-generated response string
            
        Returns:
            Filtered response string
        """
        # Implement actual filtering logic here
        # This could include checking for forbidden patterns, PII, etc.
        
        # Simple placeholders for demonstration
        sensitive_patterns = [
            "social security",
            "credit card number",
            "password",
            "pin code",
            "account number",
            "routing number"
        ]
        
        filtered_response = response
        for pattern in sensitive_patterns:
            if pattern.lower() in filtered_response.lower():
                filtered_response = filtered_response.replace(pattern, "[REDACTED]")
        
        return filtered_response
    
    def handle_jailbreaking(self, query: str) -> bool:
        """
        Detect potential jailbreak or prompt injection attempts.
        
        Args:
            query: The user's query string
            
        Returns:
            True if a potential jailbreak is detected, False otherwise
        """
        # Implement jailbreak detection logic
        jailbreak_patterns = [
            "ignore previous instructions",
            "bypass",
            "override",
            "disregard",
            "ignore your training",
            "forget your guidelines",
            "act as if",
            "pretend you are",
            "you're no longer",
            "ignore all rules"
        ]
        
        lower_query = query.lower()
        for pattern in jailbreak_patterns:
            if pattern in lower_query:
                return True
        
        return False
    
    def is_out_of_domain(self, query: str) -> bool:
        """
        Check if a query is outside the domain of banking services.
        
        Args:
            query: The user's query string
            
        Returns:
            True if the query is likely out of domain, False otherwise
        """
        # This is a simple implementation that could be improved with more sophisticated approaches
        banking_keywords = [
            # Original keywords
            "card", "interest", "balance", "statement", "transaction", "branch", "atm",
            "bank", "finance", "payment", "fund", "money", "check", "saving", "investment",
            "mortgage", "rate", "fee", "charge", "online", "mobile", "app", "password",
            "pin", "login", "security", "otp", "NUST", "customer","global ranking","admission","undergraduate","international","sports","facilities","faculty","help","department","Library",
            
            # Account types
            "account", "current", "business", "premium", "remittance", "PakWatan",
            "Value Plus", "Value Premium", "Flour Mill",
            
            # Banking services
            "transfer", "withdrawal", "deposit", "cheque", "banker's cheque", "debit card",
            "credit card", "internet banking", "SMS alerts", "e-statement", "fund transfer",
            
            # Financial products and services
            "loan", "finance", "Kamyab Jawan", "insurance", "credit", "debit",
            "KIBOR", "processing fee", "documentation", "legal charges",
            
            # Banking terminology
            "initial deposit", "minimum balance", "monthly average balance", "SOC",
            "issuance", "facility", "services", "cash management", "salary processing",
            
            # Bank-specific terms
            "VPBA", "NADRA", "NMC", "NFMF", "PWRA", "beneficiary"
        ]
        
        # Check if any banking keywords are in the query
        lower_query = query.lower()
        for keyword in banking_keywords:
            if keyword.lower() in lower_query:
                return False
        
        # If no banking keywords found, it might be out of domain
        return True
    
    def handle_out_of_domain_query(self, query: str) -> str:
        """
        Generate a response for an out-of-domain query.
        
        Args:
            query: The user's query string
            
        Returns:
            Response string for out-of-domain query
        """
        return (
            "I'm a Finbot assistant, designed to help with banking-related inquiries. "
            "It seems your question is not related to Finbot services. "
            "I'd be happy to help with questions about accounts, transfers, loans, "
            "credit cards, or other banking products and services."
        )