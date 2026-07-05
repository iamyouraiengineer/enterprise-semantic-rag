"""
src/ingestion package
Document loading and chunking pipeline.
"""

from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.chunker import RecursiveCharacterTextSplitter

__all__ = ["DocumentLoader", "Document", "RecursiveCharacterTextSplitter"]