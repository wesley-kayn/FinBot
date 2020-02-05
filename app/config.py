import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# LLM Service Configuration
LLM_SERVICE = os.getenv("LLM_SERVICE", "huggingface")  # Changed default to huggingface
MODEL = os.getenv("MODEL", "mistralai/Mistral-7B-Instruct-v0.3")  # Using Mistral v0.3
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")  # Changed to Mistral v0.3

# New embedding model configuration
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # BGE-M3 model name

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "data")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")

# Chunking configuration
MAX_CHUNK_SIZE = 1000  # Maximum characters per chunk
MIN_CHUNK_SIZE = 100   # Minimum characters per chunk
OVERLAP_SIZE = 50      # Number of characters to overlap between chunks

# Ollama Configuration (kept for backwards compatibility)
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER", "http://localhost:11434")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# App Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
APP_PORT = int(os.getenv("APP_PORT", 5014))

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)