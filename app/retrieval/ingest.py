import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.core.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_PATH,
    RAW_DATA_DIR,
    SUPPORTED_EXTENSIONS
)
from app.core.logging import logger
from app.llm.embeddings import get_huggingface_embedding_model

load_dotenv()


def get_all_raw_files() -> list[Path]:
    """
    Scans RAW_DATA_DIR recursively for all supported file formats.
    """
    data_path = Path(RAW_DATA_DIR)
    if not data_path.exists():
        logger.warning(f"Data folder not found: {RAW_DATA_DIR}. Creating directory.")
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(list(data_path.rglob(f"*{ext}")))

    return files


def load_single_document(file_path: Path) -> list:
    """
    Loads a single document based on its file extension.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext in [".md", ".txt"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif ext == ".html":
            loader = UnstructuredHTMLLoader(str(file_path))
        else:
            logger.warning(f"Unsupported extension '{ext}' for file {file_path.name}. Skipping.")
            return []

        return loader.load()
    except Exception as e:
        logger.error(f"Error loading document {file_path.name}: {e}")
        return []


def load_all_documents() -> list:
    """
    Loads all raw documents from RAW_DATA_DIR and attaches metadata.
    """
    all_documents = []
    raw_files = get_all_raw_files()

    if not raw_files:
        logger.info(f"No raw files found in {RAW_DATA_DIR}. Please add technical docs.")
        return []

    logger.info(f"Found {len(raw_files)} documents to process.")

    for file_path in raw_files:
        logger.info(f"Loading document: {file_path.name}")
        docs = load_single_document(file_path)
        doc_id = str(uuid.uuid4())

        for idx, doc in enumerate(docs, start=1):
            doc.metadata.update({
                "source_file": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "document_id": doc_id,
                "page_number": doc.metadata.get("page", idx)
            })

        all_documents.extend(docs)
        logger.info(f"Loaded {len(docs)} pages/sections from {file_path.name}")

    return all_documents


def split_documents(documents: list) -> list:
    """
    Splits documents into recursive character text chunks.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "```", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    logger.info(f"Split {len(documents)} document pages into {len(chunks)} text chunks.")
    return chunks


def build_and_save_faiss_index(chunks: list) -> None:
    """
    Creates and saves a FAISS vector store using HuggingFace embeddings.
    """
    if not chunks:
        logger.warning("No chunks available to build FAISS index.")
        return

    embedding_model = get_huggingface_embedding_model()

    if os.path.exists(FAISS_PATH):
        logger.info("Clearing previous FAISS index...")
        shutil.rmtree(FAISS_PATH)

    logger.info("Building new FAISS vector index...")
    db = FAISS.from_documents(chunks, embedding_model)

    Path(FAISS_PATH).mkdir(parents=True, exist_ok=True)
    db.save_local(FAISS_PATH)
    logger.info(f"FAISS index successfully saved to: {FAISS_PATH}")


def ingest() -> None:
    """
    Main ingestion pipeline execution function.
    """
    logger.info("Starting Document Ingestion Pipeline ...")
    documents = load_all_documents()
    chunks = split_documents(documents)
    build_and_save_faiss_index(chunks)
    logger.info("Document Ingestion Pipeline completed successfully.")


if __name__ == "__main__":
    ingest()