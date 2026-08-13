from functools import lru_cache
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_gemini_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Returns Google Gemini native embedding model instance.
    """
    model_name = settings.models.gemini_embedding_model_name
    logger.info(f"Initializing Gemini Embeddings model: {model_name}")
    return GoogleGenerativeAIEmbeddings(
        model=model_name
    )


@lru_cache(maxsize=1)
def get_huggingface_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns HuggingFace embedding model instance (default: BAAI/bge-base-en-v1.5).
    """
    model_name = settings.models.huggingface_embedding_model_name
    logger.info(f"Initializing HuggingFace Embeddings model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },
    )
