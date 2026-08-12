from pathlib import Path
from typing import Dict, Type

from app.core.logging import logger
from app.ingestion.loaders.base import BaseLoaderAdapter
from app.ingestion.loaders.document_loaders import (
    CodeLoaderAdapter,
    DocxLoaderAdapter,
    HTMLLoaderAdapter,
    MarkdownLoaderAdapter,
    PDFLoaderAdapter,
    StructuredDataLoaderAdapter,
    TextLoaderAdapter,
)


class LoaderFactory:
    """
    Registry and Factory for instantiating document loader adapters based on file extension.
    Supports runtime registration of new file formats.
    """

    _registry: Dict[str, Type[BaseLoaderAdapter]] = {
        ".pdf": PDFLoaderAdapter,
        ".md": MarkdownLoaderAdapter,
        ".markdown": MarkdownLoaderAdapter,
        ".html": HTMLLoaderAdapter,
        ".htm": HTMLLoaderAdapter,
        ".txt": TextLoaderAdapter,
        ".docx": DocxLoaderAdapter,
        ".py": CodeLoaderAdapter,
        ".js": CodeLoaderAdapter,
        ".ts": CodeLoaderAdapter,
        ".java": CodeLoaderAdapter,
        ".cpp": CodeLoaderAdapter,
        ".c": CodeLoaderAdapter,
        ".go": CodeLoaderAdapter,
        ".rs": CodeLoaderAdapter,
        ".json": StructuredDataLoaderAdapter,
        ".csv": StructuredDataLoaderAdapter,
        ".yaml": StructuredDataLoaderAdapter,
        ".yml": StructuredDataLoaderAdapter,
    }

    @classmethod
    def register_loader(cls, extension: str, loader_cls: Type[BaseLoaderAdapter]) -> None:
        """Allows registering custom loader adapters for new extensions at runtime."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        cls._registry[ext] = loader_cls
        logger.info(f"Registered custom loader '{loader_cls.__name__}' for extension '{ext}'")

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoaderAdapter:
        """
        Retrieves the appropriate loader adapter for a given file.
        Falls back to TextLoaderAdapter if extension is unregistered.
        """
        ext = file_path.suffix.lower()
        loader_cls = cls._registry.get(ext, TextLoaderAdapter)
        logger.debug(f"Selected loader '{loader_cls.__name__}' for file '{file_path.name}' ({ext})")
        return loader_cls()
