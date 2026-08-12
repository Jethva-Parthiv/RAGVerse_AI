import os
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

from app.ingestion.chunking.factory import ChunkerFactory
from app.ingestion.chunking.strategies import (
    CharacterChunker,
    CodeLanguageChunker,
    MarkdownHeaderChunker,
    RecursiveChunker,
    TokenChunker,
)
from app.ingestion.loaders.document_loaders import (
    CodeLoaderAdapter,
    MarkdownLoaderAdapter,
    TextLoaderAdapter,
)
from app.ingestion.loaders.factory import LoaderFactory
from app.ingestion.processor import IngestionProcessor


def test_loader_factory_resolution():
    md_loader = LoaderFactory.get_loader(Path("test_doc.md"))
    assert isinstance(md_loader, MarkdownLoaderAdapter)

    py_loader = LoaderFactory.get_loader(Path("script.py"))
    assert isinstance(py_loader, CodeLoaderAdapter)

    unknown_loader = LoaderFactory.get_loader(Path("unknown_file.customext"))
    assert isinstance(unknown_loader, TextLoaderAdapter)


def test_chunker_factory_auto_selection():
    md_chunker = ChunkerFactory.get_chunker(file_type=".md")
    assert isinstance(md_chunker, MarkdownHeaderChunker)

    py_chunker = ChunkerFactory.get_chunker(file_type=".py", programming_language="python")
    assert isinstance(py_chunker, CodeLanguageChunker)
    assert py_chunker.strategy_name == "code_python"

    rec_chunker = ChunkerFactory.get_chunker(file_type=".pdf")
    assert isinstance(rec_chunker, RecursiveChunker)


def test_explicit_chunker_selection():
    token_chunker = ChunkerFactory.get_chunker(strategy_name="token")
    assert isinstance(token_chunker, TokenChunker)

    char_chunker = ChunkerFactory.get_chunker(strategy_name="character")
    assert isinstance(char_chunker, CharacterChunker)


def test_chunking_metadata_enrichment():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    raw_doc = Document(
        page_content="RAGVerse_AI is an enterprise technical RAG framework built with LangChain and LangGraph.",
        metadata={"source_file": "test.txt", "file_hash": "dummyhash"},
    )
    chunks = chunker.split([raw_doc])

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == idx
        assert chunk.metadata["chunking_strategy"] == "recursive_character"
        assert chunk.metadata["source_file"] == "test.txt"
        assert chunk.metadata["file_hash"] == "dummyhash"


def test_processor_single_file(tmp_path: Path):
    sample_file = tmp_path / "sample_doc.md"
    sample_file.write_text(
        "# Section 1\nThis is paragraph one.\n\n## Section 2\nThis is paragraph two.",
        encoding="utf-8",
    )

    processor = IngestionProcessor(output_index_path=str(tmp_path / "faiss_index"))
    chunks = processor.process_file(sample_file)

    assert len(chunks) > 0
    assert "source_file" in chunks[0].metadata
    assert chunks[0].metadata["source_file"] == "sample_doc.md"
    assert "file_hash" in chunks[0].metadata
    assert "ingested_at_utc" in chunks[0].metadata
