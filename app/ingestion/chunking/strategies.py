from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.chunking.base import BaseChunkerStrategy


class RecursiveChunker(BaseChunkerStrategy):
    """General-purpose recursive character text splitter."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "```", ". ", " ", ""],
        )

    @property
    def strategy_name(self) -> str:
        return "recursive_character"

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        chunks = self.splitter.split_documents(documents)
        return self._enrich_chunks(chunks)


class MarkdownHeaderChunker(BaseChunkerStrategy):
    """Header-aware splitter for Markdown documentation."""

    HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @property
    def strategy_name(self) -> str:
        return "markdown_header"

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        final_chunks: List[Document] = []

        for doc in documents:
            header_split_docs = self.header_splitter.split_text(doc.page_content)
            for split_doc in header_split_docs:
                merged_metadata = {**doc.metadata, **split_doc.metadata}
                sub_chunks = self.sub_splitter.split_documents([
                    Document(page_content=split_doc.page_content, metadata=merged_metadata)
                ])
                final_chunks.extend(sub_chunks)

        return self._enrich_chunks(final_chunks)


class CodeLanguageChunker(BaseChunkerStrategy):
    """Syntax-aware code text splitter (AST / Language syntax aware)."""

    LANGUAGE_ENUM_MAP = {
        "python": Language.PYTHON,
        "javascript": Language.JS,
        "typescript": Language.TS,
        "cpp": Language.CPP,
        "c": Language.C,
        "go": Language.GO,
        "rust": Language.RUST,
        "html": Language.HTML,
    }

    def __init__(
        self,
        language: str = "python",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.language_str = language
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        lang_enum = self.LANGUAGE_ENUM_MAP.get(language.lower(), Language.PYTHON)
        
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @property
    def strategy_name(self) -> str:
        return f"code_{self.language_str}"

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        chunks = self.splitter.split_documents(documents)
        return self._enrich_chunks(chunks)


class TokenChunker(BaseChunkerStrategy):
    """Token-based text splitter for strict LLM token window constraints."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @property
    def strategy_name(self) -> str:
        return "token_bounded"

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        chunks = self.splitter.split_documents(documents)
        return self._enrich_chunks(chunks)


class CharacterChunker(BaseChunkerStrategy):
    """Fixed-character text splitter."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        self.splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
        )

    @property
    def strategy_name(self) -> str:
        return "character_fixed"

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        chunks = self.splitter.split_documents(documents)
        return self._enrich_chunks(chunks)
