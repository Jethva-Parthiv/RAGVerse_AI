from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseChunkerStrategy(ABC):
    """
    Abstract Base Class for all document chunking / text-splitting strategies.
    Standardizes chunk enrichment and strategy naming.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Returns the identifier name for this chunking strategy."""
        pass

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks.
        """
        pass

    def _enrich_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Enriches each chunk with chunk_id, strategy_name, and total count metadata.
        """
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["total_chunks_in_batch"] = total
            chunk.metadata["chunking_strategy"] = self.strategy_name
        return chunks
