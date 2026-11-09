"""Vector retrieval module using local embeddings and FAISS index."""

from dataclasses import dataclass
from typing import Any

from app.core.logging_config import logger
from rag.embeddings import EmbeddingService
from rag.vector_store import FAISSVectorStore


# Cosine similarity below this is treated as "not relevant" (MiniLM: related ~0.3-0.6).
MIN_RELEVANCE_SCORE = 0.20


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with similarity score."""
    text: str
    source_file: str
    page_number: int
    chunk_id: str
    score: float
    document_id: str = ""
    chunk_index: int = 1


class VectorRetriever:
    """Retriever utilizing EmbeddingService and FAISSVectorStore for semantic search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self, question: str, top_k: int = 3, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """Perform semantic similarity search for a query question.

        Args:
            question: Search query.
            top_k: Number of chunks to retrieve.

        Returns:
            List of RetrievedChunk instances ordered by cosine similarity score descending.
        """
        if not question or not question.strip():
            return []

        if self.vector_store.total_vectors == 0:
            logger.info("Vector store is empty, returning empty search results.")
            return []

        # Generate query embedding
        query_vec = self.embedding_service.embed_query(question)

        # Search FAISS index
        raw_results = self.vector_store.search(query_vec, top_k=top_k)

        threshold = MIN_RELEVANCE_SCORE if min_score is None else min_score
        results: list[RetrievedChunk] = []
        for chunk_dict, score in raw_results:
            if score < threshold:
                continue
            results.append(
                RetrievedChunk(
                    text=chunk_dict.get("text", ""),
                    source_file=chunk_dict.get("source_file", ""),
                    page_number=chunk_dict.get("page_number", 1),
                    chunk_id=chunk_dict.get("chunk_id", ""),
                    document_id=chunk_dict.get("document_id", ""),
                    chunk_index=chunk_dict.get("chunk_index", 1),
                    score=score,
                )
            )

        if not results and raw_results:
            logger.info(
                f"No chunk passed relevance threshold {threshold:.2f}; "
                f"best score was {max(score for _, score in raw_results):.3f}"
            )
        return results
