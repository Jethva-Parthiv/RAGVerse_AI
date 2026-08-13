from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.logging import logger
from app.llm.embeddings import get_huggingface_embedding_model


def get_retriever():
    """
    Returns a standard FAISS vectorstore retriever.
    """
    faiss_path = settings.vector_db.faiss_path
    top_k = settings.rag.top_k_results
    faiss_dir = Path(faiss_path)
    embedding_model = get_huggingface_embedding_model()

    if not faiss_dir.exists() or not (faiss_dir / "index.faiss").exists():
        logger.warning(f"FAISS index not found at '{faiss_path}'. Creating empty vectorstore.")
        # Create empty FAISS store to avoid crash
        db = FAISS.from_texts(["Initialization placeholder document."], embedding_model)
    else:
        db = FAISS.load_local(
            faiss_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
