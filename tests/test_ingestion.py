"""
tests/test_ingestion.py
Unit tests for document loading and chunking.

Design decisions:
- All file I/O uses pytest's tmp_path fixture for hermetic, self-contained tests.
- We test edge cases: unsupported formats, missing files, empty text, and overlap.
"""

from pathlib import Path

import pytest

from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.chunker import RecursiveCharacterTextSplitter


# =============================================================================
# DocumentLoader Tests
# =============================================================================
class TestDocumentLoader:
    def test_load_unsupported_format_returns_empty(self, tmp_path: Path) -> None:
        """An .xyz file must be rejected gracefully with an empty list."""
        path = tmp_path / "file.xyz"
        path.write_text("hello")
        loader = DocumentLoader()
        docs = loader.load_file(path)
        assert docs == []

    def test_load_missing_file_returns_empty(self) -> None:
        """A non-existent path must not raise an exception."""
        loader = DocumentLoader()
        docs = loader.load_file("/tmp/nonexistent_file_12345.pdf")
        assert docs == []

    def test_load_txt(self, tmp_path: Path) -> None:
        """TXT files must load as a single Document with correct metadata."""
        path = tmp_path / "test.txt"
        path.write_text("Hello world.\nThis is a test.")
        loader = DocumentLoader()
        docs = loader.load_file(path)
        assert len(docs) == 1
        assert docs[0].text == "Hello world.\nThis is a test."
        assert docs[0].source == str(path)
        assert docs[0].metadata["type"] == "txt"

    def test_load_directory_recursive(self, tmp_path: Path) -> None:
        """load_directory must discover files in nested subdirectories."""
        (tmp_path / "a.txt").write_text("doc a")
        (tmp_path / "b.txt").write_text("doc b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("doc c")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)
        assert len(docs) == 3
        sources = {d.source for d in docs}
        assert str(tmp_path / "a.txt") in sources
        assert str(tmp_path / "b.txt") in sources
        assert str(sub / "c.txt") in sources


# =============================================================================
# RecursiveCharacterTextSplitter Tests
# =============================================================================
class TestRecursiveCharacterTextSplitter:
    def test_short_text_no_split(self) -> None:
        """Text shorter than chunk_size must remain a single chunk."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_empty_text(self) -> None:
        """Empty string must return an empty list."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_text("")
        assert chunks == []

    def test_splits_on_separator(self) -> None:
        """
        Long text with paragraph breaks must split into multiple chunks,
        all respecting chunk_size.
        """
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= 20 for c in chunks)

    def test_overlap_present(self) -> None:
        """
        Overlap must be visible: the second chunk should start with the tail
        of the first chunk.
        """
        text = "a" * 50 + "\n\n" + "b" * 50
        splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        # The second chunk should start with the last 10 chars of the first chunk
        assert chunks[1].startswith(chunks[0][-10:])

    def test_chunk_documents_preserves_metadata(self) -> None:
        """split_documents must preserve source and inject chunk_index."""
        docs = [
            Document(
                text="Hello world this is a test",
                source="test.txt",
                metadata={"type": "txt"},
            )
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 0
        assert chunks[0].source == "test.txt"
        assert chunks[0].metadata["type"] == "txt"
        assert "chunk_index" in chunks[0].metadata
        assert chunks[0].metadata["chunk_index"] == 0

    def test_invalid_overlap_raises(self) -> None:
        """overlap >= chunk_size is a business rule violation."""
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)