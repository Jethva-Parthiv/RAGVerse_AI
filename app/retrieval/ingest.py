import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.core.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_PATH
)

from app.llm.embeddings import (
    get_gemini_embedding_model
)


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"


def load_pdf(file_path: str):
    """
    Load PDF documents.
    """

    loader = PyPDFLoader(file_path)

    return loader.load()


def split_documents(documents):
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


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
    Create new FAISS DB or update existing one.
    """

    embedding_model = get_gemini_embedding_model()

    texts, embeddings = create_embeddings(
        chunks,
        embedding_model
    )

    # ==========================================
    # LOAD EXISTING VECTOR STORE
    # ==========================================

    if os.path.exists(FAISS_PATH):

        print("\nLoading existing FAISS index...")

        db = FAISS.load_local(
            FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        print("Appending new vectors...")

        db.add_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=[
                chunk.metadata
                for chunk in chunks
            ]
        )

    # ==========================================
    # CREATE NEW VECTOR STORE
    # ==========================================

    else:

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


def save_vector_store(db):
    """
    Save FAISS index locally.
    """

    Path(FAISS_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    db.save_local(FAISS_PATH)


def ingest(file_path: str):
    """
    Complete ingestion pipeline.
    """

    print("\nLoading PDF...")

    documents = load_pdf(file_path)

    print(f"Loaded {len(documents)} pages")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # ==========================================
    # ADD SOURCE METADATA
    # ==========================================

    file_name = Path(file_path).name

    for chunk in chunks:
        chunk.metadata["source_file"] = file_name

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

    pdf_path = input(
        "\nEnter PDF path: "
    ).strip()

    if not pdf_path:
        print("\nPlease provide a PDF path.")

    elif not os.path.exists(pdf_path):
        print("\nInvalid PDF path.")

    elif not pdf_path.endswith(".pdf"):
        print("\nOnly PDF files are supported.")

    else:
        ingest(pdf_path)