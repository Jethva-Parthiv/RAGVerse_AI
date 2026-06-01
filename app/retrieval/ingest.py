import shutil
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.core.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_PATH,
    RAW_DATA_DIR
)

from app.llm.embeddings import (
    get_gemini_embedding_model
)


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ==========================================
# LOAD PDF FILES
# ==========================================

def get_pdf_files():
    """
    Get all PDF files from data directory.
    """

    data_path = Path(RAW_DATA_DIR)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder not found: {RAW_DATA_DIR}"
        )

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            "No PDF files found in data folder."
        )

    return pdf_files


# ==========================================
# LOAD DOCUMENTS
# ==========================================

def load_pdf(file_path: str):
    """
    Load PDF documents.
    """

    loader = PyPDFLoader(file_path)

    return loader.load()


def load_all_documents():
    """
    Load all PDF documents.
    """

    all_documents = []

    pdf_files = get_pdf_files()

    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        documents = load_pdf(str(pdf_file))

        # ==========================================
        # ADD DOCUMENT METADATA
        # ==========================================

        document_id = str(uuid.uuid4())

        for page_number, doc in enumerate(documents, start=1):

            doc.metadata.update({
                "source_file": pdf_file.name,
                "document_id": document_id,
                "page_number": page_number
            })

            # doc.metadata.update({
            #     "source": pdf_file.name,
            #     "document_id": document_id,
            #     "page": page_number
            # })

        all_documents.extend(documents)

        print(f"Loaded {len(documents)} pages")

    return all_documents


# ==========================================
# SPLIT DOCUMENTS
# ==========================================

def split_documents(documents):
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    # ==========================================
    # ADD CHUNK METADATA
    # ==========================================

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    return chunks


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

def create_embeddings(chunks, embedding_model):
    """
    Create embeddings for all chunks.
    """

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    print("\nCreating embeddings...")

    embeddings = []

    for index, text in enumerate(texts, start=1):

        vector = embedding_model.embed_query(text)

        embeddings.append(vector)

        print(f"Embedded chunk {index}/{len(texts)}")

    if len(texts) != len(embeddings):
        raise ValueError(
            "Mismatch between texts and embeddings."
        )

    return texts, embeddings




def create_or_update_vector_store(chunks):
    """
    Always create a fresh FAISS index.
    Deletes old index if it exists.
    """

    embedding_model = get_gemini_embedding_model()

    texts, embeddings = create_embeddings(
        chunks,
        embedding_model
    )

    # ==========================================
    # DELETE OLD FAISS INDEX
    # ==========================================

    if os.path.exists(FAISS_PATH):

        print("\nDeleting existing FAISS index...")

        shutil.rmtree(FAISS_PATH)

    # ==========================================
    # CREATE NEW VECTOR STORE
    # ==========================================

    print("\nCreating new FAISS index...")

    db = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=embedding_model,
        metadatas=[
            chunk.metadata
            for chunk in chunks
        ]
    )

    return db


# ==========================================
# SAVE VECTOR STORE
# ==========================================

def save_vector_store(db):
    """
    Save FAISS index locally.
    """

    Path(FAISS_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    db.save_local(FAISS_PATH)


# ==========================================
# INGESTION PIPELINE
# ==========================================

def ingest():
    """
    Complete ingestion pipeline.
    """

    print("\nLoading PDF documents...")

    documents = load_all_documents()

    print(f"\nTotal pages loaded: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # ==========================================
    # VECTOR STORE
    # ==========================================

    db = create_or_update_vector_store(chunks)

    # ==========================================
    # SAVE VECTOR STORE
    # ==========================================

    print("\nSaving vector database...")

    save_vector_store(db)

    print("\nFAISS index updated successfully.")
    print(f"Saved at: {FAISS_PATH}")


if __name__ == "__main__":

    ingest()