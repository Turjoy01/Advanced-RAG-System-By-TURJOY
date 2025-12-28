"""
Utility functions for the RAG system
"""

import os
import re
import logging
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_dir / log_file))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(log_dir / f"rag_system_{timestamp}.log"))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from external libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key:
        return False
    
    # Basic format validation
    if not api_key.startswith("sk-"):
        return False
    
    if len(api_key) < 20:
        return False
    
    return True

def format_sources(chunks: List[Dict]) -> List[Dict]:
    """Format source information from retrieved chunks"""
    
    sources = []
    
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        
        source = {
            'title': metadata.get('doc_title', 'Unknown Document'),
            'page': metadata.get('page_number', 'Unknown'),
            'section': metadata.get('section', 'content'),
            'score': chunk.get('final_score', chunk.get('score', 0)),
            'content_preview': chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
        }
        
        sources.append(source)
    
    return sources

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"]', '', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    
    if len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can find a good word boundary
        return truncated[:last_space] + suffix
    else:
        return truncated + suffix

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters"""
    
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Trim and ensure not empty
    safe_name = safe_name.strip('_')
    
    if not safe_name:
        safe_name = "document"
    
    return safe_name

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def validate_document_path(file_path: str) -> bool:
    """Validate document file path"""
    
    if not file_path:
        return False
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return False
    
    # Check if it's a file (not directory)
    if not path.is_file():
        return False
    
    # Check file extension
    allowed_extensions = {'.pdf', '.txt', '.docx', '.doc'}
    if path.suffix.lower() not in allowed_extensions:
        return False
    
    return True

class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.description} completed in {duration:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0