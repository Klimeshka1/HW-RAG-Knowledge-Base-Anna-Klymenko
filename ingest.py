"""
ingest.py — PDF processing pipeline for Financial Literacy RAG app.

Loads PDFs from data/books/, chunks text, creates embeddings via OpenAI,
and persists them to ChromaDB.

Usage:
    python ingest.py
    python ingest.py --reset   # wipe chroma_db and reingest
"""

import os
import sys
import argparse
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BOOKS_DIR = Path("data/books")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_literacy"

CHUNK_SIZE = 500        # tokens approximate — splitter uses characters, ~4 chars/token
CHUNK_OVERLAP = 50
CHUNK_SIZE_CHARS = CHUNK_SIZE * 4
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * 4

MIN_PAGE_CHARS = 50     # pages with fewer chars are considered unreadable (scanned)


def clean_filename(filename: str) -> str:
    """Turn a filename into a readable book title."""
    name = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", name)
    return name.strip().title()


def load_pdf(pdf_path: Path) -> list:
    """Load a PDF and return a list of LangChain Document objects."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        return pages
    except Exception as e:
        print(f"  [ERROR] Could not load {pdf_path.name}: {e}")
        return []


def filter_pages(pages: list, book_title: str) -> list:
    """Drop pages with too little text (likely scanned images) and warn."""
    good, skipped = [], 0
    for doc in pages:
        text = doc.page_content.strip()
        if len(text) < MIN_PAGE_CHARS:
            skipped += 1
        else:
            good.append(doc)
    if skipped:
        print(f"  [WARN] Skipped {skipped} low-text page(s) in '{book_title}' (likely scanned)")
    return good


def ingest_books(reset: bool = False) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    if not BOOKS_DIR.exists() or not list(BOOKS_DIR.glob("*.pdf")):
        print(f"[ERROR] No PDF files found in {BOOKS_DIR}/")
        print("Place your PDF books there and re-run.")
        sys.exit(1)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    # Wipe existing DB if requested
    if reset and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("[INFO] Existing ChromaDB wiped.")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdf_files = sorted(BOOKS_DIR.glob("*.pdf"))
    total_chunks = 0

    for pdf_path in pdf_files:
        book_title = clean_filename(pdf_path.name)
        print(f"\nProcessing: {book_title}")

        pages = load_pdf(pdf_path)
        if not pages:
            continue

        pages = filter_pages(pages, book_title)
        if not pages:
            print(f"  [SKIP] No readable text found in '{book_title}'")
            continue

        # Inject clean book title into metadata
        for doc in pages:
            doc.metadata["book_title"] = book_title
            # PyPDFLoader sets 'page' (0-indexed); normalise to 1-indexed
            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1

        chunks = splitter.split_documents(pages)
        print(f"  Pages readable: {len(pages)} | Chunks created: {len(chunks)}")

        if chunks:
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  Stored {len(chunks)} chunks to ChromaDB.")

    print(f"\n[DONE] Total chunks stored: {total_chunks}")
    print(f"ChromaDB saved to: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB before ingesting",
    )
    args = parser.parse_args()
    ingest_books(reset=args.reset)
