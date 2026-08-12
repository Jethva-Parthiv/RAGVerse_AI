from typing import Any, Dict, Optional, Type

from app.core.logging import logger
from app.ingestion.chunking.base import BaseChunkerStrategy
from app.ingestion.chunking.strategies import (
    CharacterChunker,
    CodeLanguageChunker,
    MarkdownHeaderChunker,
    RecursiveChunker,
    TokenChunker,
)


class ChunkerFactory:
    """
    Factory for selecting and instantiating document chunking strategies.
    Supports auto-selection based on document type/metadata or explicit strategy names.
    """

    _registry: Dict[str, Type[BaseChunkerStrategy]] = {
        "recursive": RecursiveChunker,
        "markdown": MarkdownHeaderChunker,
        "code": CodeLanguageChunker,
        "token": TokenChunker,
        "character": CharacterChunker,
    }

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: Type[BaseChunkerStrategy]) -> None:
        """Registers a custom chunking strategy at runtime."""
        cls._registry[name.lower()] = strategy_cls
        logger.info(f"Registered custom chunking strategy '{name}' -> {strategy_cls.__name__}")

    @classmethod
    def get_chunker(
        cls,
        strategy_name: Optional[str] = None,
        file_type: Optional[str] = None,
        programming_language: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChunkerStrategy:
        """
        Intelligently resolves and instantiates a chunker strategy.
        """
        # Explicit strategy name requested
        if strategy_name and strategy_name.lower() in cls._registry:
            strat_cls = cls._registry[strategy_name.lower()]
            if strat_cls == CodeLanguageChunker and programming_language:
                return CodeLanguageChunker(language=programming_language, **kwargs)
            return strat_cls(**kwargs)

        # Auto-selection logic based on file type or programming language metadata
        if file_type in [".md", ".markdown"]:
            logger.debug(f"Auto-selected MarkdownHeaderChunker for file type '{file_type}'")
            return MarkdownHeaderChunker(**kwargs)

        if programming_language or file_type in [".py", ".js", ".ts", ".cpp", ".c", ".go", ".rs"]:
            lang = programming_language or "python"
            logger.debug(f"Auto-selected CodeLanguageChunker ({lang}) for file type '{file_type}'")
            return CodeLanguageChunker(language=lang, **kwargs)

        # Fallback to RecursiveChunker
        logger.debug(f"Defaulting to RecursiveChunker for file type '{file_type}'")
        return RecursiveChunker(**kwargs)
