import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.chunking.factory import ChunkerFactory
from app.ingestion.loaders.factory import LoaderFactory
from app.llm.embeddings import get_huggingface_embedding_model


class IngestionProcessor:
    """
    Industry-grade ingestion processor that orchestrates scanning, loading,
    chunking, metadata enrichment, and vector database indexing.
    """

    def __init__(self, output_index_path: Optional[str] = None):
        self.output_index_path = output_index_path or settings.vector_db.faiss_path

    def process_file(
        self,
        file_path: Path | str,
        strategy_name: Optional[str] = None,
    ) -> List:
        """
        Processes a single file: resolves loader adapter, loads content,
        selects chunking strategy, splits content into enriched chunks.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        logger.info(f"Processing file: {path.name}")
        loader = LoaderFactory.get_loader(path)
        raw_docs = loader.load(path)

        if not raw_docs:
            logger.warning(f"No content loaded from file {path.name}")
            return []

        doc_id = str(uuid.uuid4())
        ingested_at = datetime.now(timezone.utc).isoformat()

        # Attach high-level document identification metadata
        for doc in raw_docs:
            doc.metadata["document_id"] = doc_id
            doc.metadata["ingested_at_utc"] = ingested_at

        first_doc = raw_docs[0]
        file_type = first_doc.metadata.get("file_type", path.suffix.lower())
        prog_lang = first_doc.metadata.get("programming_language")

        chunker = ChunkerFactory.get_chunker(
            strategy_name=strategy_name,
            file_type=file_type,
            programming_language=prog_lang,
        )

        chunks = chunker.split(raw_docs)
        logger.info(f"Generated {len(chunks)} chunks for '{path.name}' using strategy '{chunker.strategy_name}'")
        return chunks

    def process_directory(
        self,
        source_dir: Optional[Path | str] = None,
        strategy_name: Optional[str] = None,
    ) -> List:
        """
        Recursively scans source directory for supported files and processes them.
        Isolates per-file errors to prevent complete pipeline failures.
        """
        target_dir = source_dir or settings.paths.raw_data_dir
        data_path = Path(target_dir).resolve()
        if not data_path.exists():
            logger.warning(f"Source directory '{data_path}' does not exist. Creating it.")
            data_path.mkdir(parents=True, exist_ok=True)
            return []

        supported_files: List[Path] = []
        for ext in settings.rag.supported_extensions:
            supported_files.extend(list(data_path.rglob(f"*{ext}")))

        if not supported_files:
            logger.info(f"No supported files found in '{data_path}'.")
            return []

        logger.info(f"Found {len(supported_files)} supported files in '{data_path}'. Starting batch processing...")

        all_chunks: List = []
        successful_count = 0
        failed_count = 0

        for file_path in supported_files:
            try:
                chunks = self.process_file(file_path, strategy_name=strategy_name)
                if chunks:
                    all_chunks.extend(chunks)
                    successful_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing file '{file_path.name}': {e}", exc_info=True)

        logger.info(
            f"Ingestion batch complete. Successfully processed: {successful_count}, Failed: {failed_count}, Total Chunks: {len(all_chunks)}"
        )
        return all_chunks

    def build_and_save_index(self, chunks: List) -> None:
        """
        Creates and saves a FAISS vector index from the ingested document chunks.
        """
        if not chunks:
            logger.warning("No chunks available to build vector index.")
            return

        logger.info("Initializing embedding model for vector store...")
        embedding_model = get_huggingface_embedding_model()

        if os.path.exists(self.output_index_path):
            logger.info(f"Removing old vector index at '{self.output_index_path}'...")
            shutil.rmtree(self.output_index_path)

        logger.info("Building FAISS vector index from document chunks...")
        db = FAISS.from_documents(chunks, embedding_model)

        Path(self.output_index_path).mkdir(parents=True, exist_ok=True)
        db.save_local(self.output_index_path)
        logger.info(f"Vector index successfully built and saved to '{self.output_index_path}'")
