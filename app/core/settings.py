"""
RAGVerse_AI Central Configuration Settings
-------------------------------------------
Pipes structured settings from `app.core.config` into convenient top-level constants
and exports the `settings` singleton instance.
"""

from app.core.config import AppConfig, get_settings, settings

# -----------------------------------
# BASE PATHS
# -----------------------------------
BASE_DIR = settings.paths.base_dir
DATA_DIR = settings.paths.data_dir
PROCESSED_DATA_DIR = settings.paths.processed_data_dir
RAW_DATA_DIR = settings.paths.raw_data_dir
EVAL_DATA_DIR = settings.paths.eval_data_dir
LOG_DIR = settings.paths.log_dir

# -----------------------------------
# MODELS
# -----------------------------------
CHAT_MODEL_NAME = settings.models.chat_model_name
GEMINI_EMBEDDING_MODEL_NAME = settings.models.gemini_embedding_model_name
HUGGINGFACE_EMBEDDING_MODEL_NAME = settings.models.huggingface_embedding_model_name

# -----------------------------------
# RAG CONFIG
# -----------------------------------
TOP_K_RESULTS = settings.rag.top_k_results
CHUNK_SIZE = settings.rag.chunk_size
CHUNK_OVERLAP = settings.rag.chunk_overlap
SUPPORTED_EXTENSIONS = settings.rag.supported_extensions

# -----------------------------------
# VECTOR DATABASE
# -----------------------------------
FAISS_PATH = settings.vector_db.faiss_path
QDRANT_HOST = settings.vector_db.qdrant_host
QDRANT_PORT = settings.vector_db.qdrant_port
QDRANT_COLLECTION_NAME = settings.vector_db.qdrant_collection_name

# -----------------------------------
# DATABASE
# -----------------------------------
DATABASE_URL = settings.database.database_url

__all__ = [
    "settings",
    "get_settings",
    "AppConfig",
    "BASE_DIR",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "RAW_DATA_DIR",
    "EVAL_DATA_DIR",
    "LOG_DIR",
    "CHAT_MODEL_NAME",
    "GEMINI_EMBEDDING_MODEL_NAME",
    "HUGGINGFACE_EMBEDDING_MODEL_NAME",
    "TOP_K_RESULTS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "SUPPORTED_EXTENSIONS",
    "FAISS_PATH",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION_NAME",
    "DATABASE_URL",
]