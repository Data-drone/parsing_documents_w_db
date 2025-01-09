
from pathlib import Path
from datetime import datetime
import hashlib

def generate_doc_id(filename: str, timestamp: str) -> str:
    """
    Generate a deterministic ID combining the filename and timestamp of processing
    
    """
    content = f"{filename}{timestamp}"
    return hashlib.sha256(content.encode()).hexdigest()

def extract_filename(dbfs_path: str) -> str:
    """
    Extracts the filename from the volume file path
    
    Example: "dbfs:/Volumes/catalog/schema/volume_name/file.pdf" become "file.pdf"
            
    """
    return Path(dbfs_path).name

def extract_datetime() -> str:
    """
    Get current datetime in ISO format
    """
    return datetime.utcnow().isoformat()