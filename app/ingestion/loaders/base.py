import hashlib
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

from app.core.logging import logger


class BaseLoaderAdapter(ABC):
    """
    Abstract Base Class for all RAGVerse document loader adapters.
    Ensures consistent loading behavior and standardized file metadata extraction.
    """

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """
        Loads documents from the given file path and returns a list of LangChain Document objects.
        """
        pass

    def extract_file_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extracts foundational metadata from the file system.
        """
        try:
            stat = file_path.stat()
            file_size = stat.st_size
            mod_time = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            
            # Compute SHA-256 hash for deduplication and tracking
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()
            
            return {
                "source_file": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_type": file_path.suffix.lower(),
                "file_size_bytes": file_size,
                "file_hash": file_hash,
                "last_modified_utc": mod_time,
            }
        except Exception as e:
            logger.warning(f"Could not extract full file metadata for {file_path.name}: {e}")
            return {
                "source_file": file_path.name,
                "file_path": str(file_path.resolve()),
                "file_type": file_path.suffix.lower(),
            }
