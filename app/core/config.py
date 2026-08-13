from functools import lru_cache
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class PathsConfig(BaseModel):
    """File system directory configuration."""
    base_dir: Path = Field(default_factory=lambda: ROOT_DIR)
    data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data")
    processed_data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "processed")
    raw_data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "raw" / "documents")
    eval_data_dir: Path = Field(default_factory=lambda: ROOT_DIR / "data" / "eval")
    log_dir: Path = Field(default_factory=lambda: ROOT_DIR / "logs")

    def create_directories(self) -> None:
        """Ensures all essential directories exist on disk."""
        for directory in [
            self.data_dir,
            self.processed_data_dir,
            self.raw_data_dir,
            self.eval_data_dir,
            self.log_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseModel):
    """LLM & Embeddings model configurations."""
    chat_model_name: str = Field(
        default_factory=lambda: os.getenv("CHAT_MODEL_NAME", "gemini-3.1-flash-lite")
    )
    gemini_embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "gemini-embedding-001")
    )
    huggingface_embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
    )


class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation hyperparameters & document parameters."""
    top_k_results: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5"))
    )
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    supported_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".md", ".markdown", ".docx", ".html", ".htm", ".txt",
            ".py", ".js", ".ts", ".json", ".csv", ".yaml", ".yml"
        ]
    )


class VectorDBConfig(BaseModel):
    """Vector database storage configuration."""
    faiss_path: str = Field(
        default_factory=lambda: str(ROOT_DIR / "data" / "processed" / "faiss_index")
    )
    qdrant_host: str = Field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "localhost")
    )
    qdrant_port: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    qdrant_collection_name: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "langchain_docs")
    )


class DatabaseConfig(BaseModel):
    """Relational database & checkpointer configuration."""
    database_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("DATABASE_URL")
    )


class AppConfig(BaseModel):
    """Central Application Configuration Registry."""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    def model_post_init(self, __context) -> None:
        """Automated setup tasks following initialization."""
        self.paths.create_directories()


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Returns a cached instance of the application configuration."""
    return AppConfig()


# Singleton instance for direct access
settings = get_settings()
