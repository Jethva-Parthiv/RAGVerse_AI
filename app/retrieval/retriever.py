from pathlib import Path
from langchain_community.vectorstores import FAISS
from app.llm.embeddings import get_huggingface_embedding_model
from app.core.settings import TOP_K_RESULTS, FAISS_PATH
from app.core.logging import logger

def get_retriever():
    """
    Returns a standard FAISS vectorstore retriever.
    """
    faiss_dir = Path(FAISS_PATH)
    embedding_model = get_huggingface_embedding_model()

    if not faiss_dir.exists() or not (faiss_dir / "index.faiss").exists():
        logger.warning(f"FAISS index not found at '{FAISS_PATH}'. Creating empty vectorstore.")
        # Create empty FAISS store to avoid crash
        db = FAISS.from_texts(["Initialization placeholder document."], embedding_model)
    else:
        db = FAISS.load_local(
            FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
