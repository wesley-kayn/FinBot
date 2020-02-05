import io
import json
import re
import csv
import os
import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import unicodedata
import html
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN")

# Model & Vector-store paths
MODEL_NAME = "BAAI/bge-m3"
VECTOR_STORE_PATH = Path("app/data/vector_store")
INDEX_PATH = VECTOR_STORE_PATH / "faiss.index"
META_PATH = VECTOR_STORE_PATH / "meta.pkl"

# Chunking configuration
MAX_CHUNK_SIZE = 1000  # Maximum characters per chunk
MIN_CHUNK_SIZE = 100   # Minimum characters per chunk
OVERLAP_SIZE = 50      # Number of characters to overlap between chunks

# Enhanced PII detection patterns
PII_PATTERNS = {
    'credit_card': r'(?:\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)',
    'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
    'phone': r'\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'account_number': r'\b\d{10,17}\b',
    'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    'address': r'\b\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way)\b',
}

# Compile all patterns
PII_REGEX = re.compile('|'.join(f'(?P<{k}>{v})' for k, v in PII_PATTERNS.items()), flags=re.I)

def normalize_text(text: str) -> str:
    """Normalize text by handling unicode, HTML entities, and whitespace."""
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_and_redact_pii(text: str) -> str:
    """Detect and redact PII from text."""
    def replace_match(match):
        pii_type = match.lastgroup
        return f"[REDACTED_{pii_type.upper()}]"
    
    return PII_REGEX.sub(replace_match, text)

def clean_text(text: str) -> str:
    """Apply comprehensive text cleaning."""
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = normalize_text(text)
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Redact PII
    text = detect_and_redact_pii(text)
    
    return text.strip()

def validate_chunk(chunk: str) -> bool:
    """Validate if a text chunk is worth processing."""
    if not chunk or not isinstance(chunk, str):
        return False
    
    # Check minimum length (e.g., 10 characters)
    if len(chunk.strip()) < 10:
        return False
    
    # Check if it's not just whitespace or special characters
    if not re.search(r'[a-zA-Z]', chunk):
        return False
    
    return True

def process_chunk(chunk: str) -> str:
    """Process a single text chunk with all preprocessing steps."""
    if not validate_chunk(chunk):
        return ""
    
    return clean_text(chunk)

def open_utf8(p: Path):
    try:
        return open(p, mode="r", encoding="utf-8")
    except UnicodeDecodeError:
        return io.StringIO(p.read_bytes().decode("utf-8", errors="replace"))

def split_into_chunks(text: str) -> Generator[str, None, None]:
    """
    Split text into overlapping chunks based on semantic boundaries and size limits.
    
    Args:
        text: The input text to split
        
    Yields:
        Chunks of text that are semantically meaningful and within size limits
    """
    # First split by paragraphs
    paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # If paragraph is too long, split it into sentences
        if len(paragraph) > MAX_CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would exceed max size, yield current chunk
                if current_size + len(sentence) > MAX_CHUNK_SIZE and current_chunk:
                    chunk = ' '.join(current_chunk)
                    if len(chunk) >= MIN_CHUNK_SIZE:
                        yield chunk
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(chunk) - OVERLAP_SIZE)
                    current_chunk = [chunk[overlap_start:]]
                    current_size = len(current_chunk[0])
                
                current_chunk.append(sentence)
                current_size += len(sentence)
        else:
            # If adding this paragraph would exceed max size, yield current chunk
            if current_size + len(paragraph) > MAX_CHUNK_SIZE and current_chunk:
                chunk = ' '.join(current_chunk)
                if len(chunk) >= MIN_CHUNK_SIZE:
                    yield chunk
                
                # Start new chunk with overlap
                overlap_start = max(0, len(chunk) - OVERLAP_SIZE)
                current_chunk = [chunk[overlap_start:]]
                current_size = len(current_chunk[0])
            
            current_chunk.append(paragraph)
            current_size += len(paragraph)
    
    # Yield the final chunk if it exists
    if current_chunk:
        chunk = ' '.join(current_chunk)
        if len(chunk) >= MIN_CHUNK_SIZE:
            yield chunk

def _read_generic(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open_utf8(path) as f:
            data = json.load(f)
        
        def flatten_json(obj, parent_key=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    yield from flatten_json(v, new_key)
            elif isinstance(obj, list):
                for v in obj:
                    yield from flatten_json(v, parent_key)
            else:
                yield f"{parent_key}: {obj}"
        
        for chunk in flatten_json(data):
            processed = process_chunk(chunk)
            if processed:
                yield processed
    elif suffix == ".csv":
        with open_utf8(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed = process_chunk(row.get("text", ""))
                if processed:
                    yield processed
    else:
        # for .txt and other files
        text = path.read_text(encoding="utf-8", errors="replace")
        # Split text into chunks and process each chunk
        for chunk in split_into_chunks(text):
            processed = process_chunk(chunk)
            if processed:
                yield processed

# Modify the build_index_from_excel function in excel_parser.py

def build_index_from_excel(excel_path, vector_store_path=None):
    """
    Process Excel file by breaking sheets into text files and building a FAISS index.
    
    Args:
        excel_path: Path to the Excel file
        vector_store_path: Path to store the vector index (default: app/data/vector_store)
    """
    from app.utils.vector_store import VectorStore  # Import here to avoid circular imports
    
    excel_path = Path(excel_path)
    excel_dir = excel_path.parent
    
    # Track processing statistics
    stats = {
        "total_sheets": 0,
        "processed_sheets": 0,
        "total_chunks": 0,
        "valid_chunks": 0
    }
    
    # Create text files for each sheet in the Excel file
    try:
        print(f"Processing Excel file: {excel_path}")
        sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
        stats["total_sheets"] = len(sheets)
        
        for sheet_name, df in sheets.items():
            try:
                txt_path = excel_dir / f"{excel_path.stem}_{sheet_name}.txt"
                df.to_csv(txt_path, index=False, sep="\t")
                stats["processed_sheets"] += 1
                print(f"Created text file for sheet: {sheet_name}")
            except Exception as e:
                print(f"Failed to convert sheet {sheet_name}: {e}")
    
    except Exception as e:
        print(f"Failed to process Excel file {excel_path}: {e}")
        return False
    
    # Initialize vector store
    vector_store = VectorStore(vector_store_path=vector_store_path or "app/data/vector_store")
    
    # Process the generated text files
    text_files = list(excel_dir.glob(f"{excel_path.stem}_*.txt"))
    print(f"Found {len(text_files)} text files to process")
    
    all_chunks = []
    
    with tqdm(text_files, desc="Processing text files", unit="file") as pbar:
        for file in pbar:
            try:
                chunks = list(_read_generic(file))
                stats["total_chunks"] += len(chunks)
                valid_chunks = [c for c in chunks if c]
                stats["valid_chunks"] += len(valid_chunks)
                
                for chunk in valid_chunks:
                    sheet_name = file.stem.replace(f"{excel_path.stem}_", "")
                    all_chunks.append({
                        "content": chunk,
                        "source": excel_path.name,
                        "sheet_name": sheet_name
                    })
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Create the vector store
    if all_chunks:
        vector_store.create_vector_store(all_chunks)
        
        print("Vector store built with the following statistics:")
        print(f" - Total sheets: {stats['total_sheets']}")
        print(f" - Processed sheets: {stats['processed_sheets']}")
        print(f" - Total chunks: {stats['total_chunks']}")
        print(f" - Valid chunks: {stats['valid_chunks']}")
        return True
    else:
        print("No valid chunks found to build vector store.")
        return False

def search_index(query, k=5):
    """
    Search the built index with a query.
    
    Args:
        query: Query string
        k: Number of results to return
        
    Returns:
        List of relevant chunks with metadata
    """
    if not INDEX_PATH.exists() or not META_PATH.exists():
        print("Vector store not found. Please build the index first.")
        return []
    
    # Load the model, index and metadata
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)
    index = faiss.read_index(str(INDEX_PATH))
    
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    
    # Create query embedding
    query_vector = model.encode(query, convert_to_numpy=True)
    query_vector = query_vector.reshape(1, -1)
    
    # Search
    scores, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(meta):
            result = meta[idx].copy() if isinstance(meta[idx], dict) else {"content": meta[idx]}
            result["score"] = float(scores[0][i])
            results.append(result)
    
    return results