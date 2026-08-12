import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document

from app.core.logging import logger
from app.ingestion.loaders.base import BaseLoaderAdapter


class PDFLoaderAdapter(BaseLoaderAdapter):
    """Adapter for PDF documents."""

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for idx, doc in enumerate(docs, start=1):
                doc.metadata.update(base_meta)
                doc.metadata["page_number"] = doc.metadata.get("page", idx)
            return docs
        except Exception as e:
            logger.error(f"PDFLoaderAdapter failed for {file_path.name}: {e}")
            return []


class MarkdownLoaderAdapter(BaseLoaderAdapter):
    """Adapter for Markdown files (.md, .markdown)."""

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update(base_meta)
            return docs
        except Exception as e:
            logger.error(f"MarkdownLoaderAdapter failed for {file_path.name}: {e}")
            return []


class HTMLLoaderAdapter(BaseLoaderAdapter):
    """Adapter for HTML documents (.html, .htm)."""

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        try:
            # Try BSHTMLLoader first for clean text extraction, fallback to Unstructured or Text
            try:
                loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")
                docs = loader.load()
            except Exception:
                try:
                    loader = UnstructuredHTMLLoader(str(file_path))
                    docs = loader.load()
                except Exception:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs = loader.load()

            for doc in docs:
                doc.metadata.update(base_meta)
            return docs
        except Exception as e:
            logger.error(f"HTMLLoaderAdapter failed for {file_path.name}: {e}")
            return []


class TextLoaderAdapter(BaseLoaderAdapter):
    """Adapter for plain text files (.txt)."""

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update(base_meta)
            return docs
        except Exception as e:
            logger.error(f"TextLoaderAdapter failed for {file_path.name}: {e}")
            return []


class DocxLoaderAdapter(BaseLoaderAdapter):
    """Adapter for Word documents (.docx)."""

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        try:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata.update(base_meta)
            return docs
        except Exception as e:
            logger.error(f"DocxLoaderAdapter failed for {file_path.name}: {e}")
            return []


class CodeLoaderAdapter(BaseLoaderAdapter):
    """Adapter for source code files (.py, .js, .ts, etc.)."""

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
    }

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        ext = file_path.suffix.lower()
        language = self.LANGUAGE_MAP.get(ext, "unknown")
        base_meta["programming_language"] = language

        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update(base_meta)
            return docs
        except Exception as e:
            logger.error(f"CodeLoaderAdapter failed for {file_path.name}: {e}")
            return []


class StructuredDataLoaderAdapter(BaseLoaderAdapter):
    """Adapter for structured files (.json, .csv, .yaml, .yml)."""

    def load(self, file_path: Path) -> List[Document]:
        base_meta = self.extract_file_metadata(file_path)
        ext = file_path.suffix.lower()

        try:
            if ext in [".json"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                text = json.dumps(content, indent=2)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            doc = Document(page_content=text, metadata=base_meta)
            return [doc]
        except Exception as e:
            logger.error(f"StructuredDataLoaderAdapter failed for {file_path.name}: {e}")
            return []
