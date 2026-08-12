from pathlib import Path

from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()


# -----------------------------------
# BASE PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw/documents"
EVAL_DATA_DIR = DATA_DIR / "eval"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for path in [DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, EVAL_DATA_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# MODELS
# -----------------------------------

CHAT_MODEL_NAME = os.getenv(
    "CHAT_MODEL_NAME",
    "gemini-3.1-flash-lite"
)

GEMINI_EMBEDDING_MODEL_NAME = os.getenv(
    "GEMINI_EMBEDDING_MODEL_NAME",
    "gemini-embedding-001"
)

HUGGINGFACE_EMBEDDING_MODEL_NAME = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL_NAME",
    "BAAI/bge-base-en-v1.5"
)

# -----------------------------------
# RAG CONFIG
# -----------------------------------

TOP_K_RESULTS = int(
    os.getenv("TOP_K_RESULTS", 5)
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 1000)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 200)
)

SUPPORTED_EXTENSIONS = [
    ".pdf", ".md", ".markdown", ".docx", ".html", ".htm", ".txt",
    ".py", ".js", ".ts", ".json", ".csv", ".yaml", ".yml"
]


# -----------------------------------
# VECTOR DATABASE (FAISS / QDRANT)
# -----------------------------------

FAISS_PATH = str(
    PROCESSED_DATA_DIR / "faiss_index"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "langchain_docs")


# -----------------------------------
# DATABASE
# -----------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)