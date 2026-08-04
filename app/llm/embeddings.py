from functools import lru_cache
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.settings import (
    GEMINI_EMBEDDING_MODEL_NAME,
    HUGGINGFACE_EMBEDDING_MODEL_NAME
)
from app.core.logging import logger

load_dotenv()


@lru_cache(maxsize=1)
def get_gemini_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Returns Google Gemini native embedding model instance.
    """
    logger.info(f"Initializing Gemini Embeddings model: {GEMINI_EMBEDDING_MODEL_NAME}")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL_NAME
    )


@lru_cache(maxsize=1)
def get_huggingface_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns HuggingFace embedding model instance (default: BAAI/bge-base-en-v1.5).
    """
    logger.info(f"Initializing HuggingFace Embeddings model: {HUGGINGFACE_EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },
    )
