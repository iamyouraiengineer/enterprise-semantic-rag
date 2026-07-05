"""
src/ingestion/chunker.py
Recursive character text splitter with configurable overlap.

Design decisions:
- We implement our own recursive splitter to demonstrate deep understanding
  of chunking mechanics, rather than blindly importing LangChain.
- Separators are ordered from coarse to fine: paragraphs -> lines -> sentences -> words.
- Overlap is implemented by prepending the tail of the previous chunk to the next.
- If overlap causes a chunk to exceed chunk_size, it is trimmed from the left.
- This preserves semantic boundaries better than naive fixed-length slicing.
"""

from typing import List

from loguru import logger

from config.settings import get_settings
from src.ingestion.loader import Document


class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks using a hierarchy of separators.
    If a chunk is too large with one separator, it falls back to the next finer one.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ):
        settings = get_settings()
        # CRITICAL FIX: use 'is not None' because 0 is a valid value for chunk_overlap.
        # '0 or 50' would incorrectly evaluate to 50.
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of loaded documents into smaller chunks.
        Preserves source and metadata, adding chunk_index.
        """
        chunked_docs: List[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for idx, chunk_text in enumerate(chunks):
                metadata = {**doc.metadata, "chunk_index": idx}
                chunked_docs.append(
                    Document(
                        text=chunk_text,
                        source=doc.source,
                        metadata=metadata,
                    )
                )
            logger.debug(
                "Chunked document | source={} | chunks={}", doc.source, len(chunks)
            )
        return chunked_docs

    def split_text(self, text: str) -> List[str]:
        """
        Public entrypoint: split a raw string into overlapping chunks.
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        # 1. Recursively split into fine-grained pieces
        fine_splits = self._recursive_split(text, self.separators[:])

        # 2. Greedily merge pieces into chunks <= chunk_size
        chunks = self._merge_splits(fine_splits)

        # 3. Apply overlap between consecutive chunks
        return self._apply_overlap(chunks)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Core recursive logic:
        1. Try splitting with the current separator.
        2. If any resulting piece is still > chunk_size, recurse with the next separator.
        3. If no separators left, hard-split by character.
        """
        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            # Final fallback: hard split by character
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        splits = text.split(separator) if separator else list(text)

        # Recurse on pieces that are still too large
        fine_splits: List[str] = []
        for piece in splits:
            if len(piece) > self.chunk_size:
                fine_splits.extend(self._recursive_split(piece, remaining_separators))
            else:
                fine_splits.append(piece)

        return fine_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Greedily merge fine-grained splits into chunks as large as possible
        without exceeding chunk_size.
        """
        chunks: List[str] = []
        current_chunk = ""

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                chunks.append(current_chunk)
                current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """Character-level split when all separators are exhausted."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Prepend the last `chunk_overlap` characters of the previous chunk
        to the current chunk. If the combined text exceeds chunk_size,
        trim from the left to fit.
        """
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = chunks[i - 1][-self.chunk_overlap :]
            combined = overlap + chunks[i]

            # Trim if overlap pushes us over chunk_size
            if len(combined) > self.chunk_size:
                combined = combined[-self.chunk_size :]

            result.append(combined)

        return result

            
     
    
