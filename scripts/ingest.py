"""
scripts/ingest.py
CLI entrypoint for document ingestion.

Usage:
    python scripts/ingest.py --file data/raw/sample.txt
    python scripts/ingest.py --dir data/raw
"""

import argparse

from config.log_config import configure_logging
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import RecursiveCharacterTextSplitter


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG pipeline"
    )
    parser.add_argument("--file", type=str, help="Path to a single file")
    parser.add_argument("--dir", type=str, help="Path to a directory of files")
    args = parser.parse_args()

    loader = DocumentLoader()
    chunker = RecursiveCharacterTextSplitter()

    if args.file:
        docs = loader.load_file(args.file)
    elif args.dir:
        docs = loader.load_directory(args.dir)
    else:
        print("Error: Provide --file or --dir")
        return

    if not docs:
        print("No documents loaded.")
        return

    chunks = chunker.split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk.source}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Text preview: {chunk.text[:200]}...")


if __name__ == "__main__":
    main()