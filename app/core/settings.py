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


# -----------------------------------
# MODELS
# -----------------------------------

CHAT_MODEL_NAME = os.getenv(
    "CHAT_MODEL_NAME",
    "gemini-2.5-flash"
)

GEMINI_EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "gemini-embedding-001"
)

HUGGINGFACE_EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
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


# -----------------------------------
# VECTOR DATABASE
# -----------------------------------

FAISS_PATH = str(
    PROCESSED_DATA_DIR / "faiss_index"
)


# -----------------------------------
# DATABASE
# -----------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


SUPPORTED_EXTENSIONS = [".pdf"]