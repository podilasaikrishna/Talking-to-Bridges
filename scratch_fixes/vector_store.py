"""Local FAISS vector store module with persistence.

Implements IndexFlatIP (cosine similarity on L2-normalized vectors)
and stores associated chunk metadata in metadata.json.
"""

import json
from pathlib import Path
from typing import Any
import faiss
import numpy as np

from app.core.config import settings
from app.core.logging_config import logger


class VectorStoreError(Exception):
    """Exception raised for vector store operations."""


class FAISSVectorStore:
    """Vector store backed by a local FAISS IndexFlatIP index and persistent metadata."""

    def __init__(self, dimension: int | None = None, db_dir: Path | str | None = None):
        self.dimension = dimension
        self.db_dir = Path(db_dir or settings.VECTOR_DB_PATH)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []

        if self.dimension is not None:
            self._init_index(self.dimension)

    def _init_index(self, dimension: int) -> None:
        """Initialize a new FAISS IndexFlatIP."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    @property
    def total_vectors(self) -> int:
        """Return the number of vectors in the store."""
        return self.index.ntotal if self.index is not None else 0

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the FAISS index.

        Args:
            chunks: List of chunk metadata dictionaries.
            embeddings: 2D numpy array of float32 embeddings (N, dimension).
        """
        if len(chunks) == 0:
            return

        if embeddings.ndim != 2:
            raise VectorStoreError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

        num_vectors, dim = embeddings.shape

        if len(chunks) != num_vectors:
            raise VectorStoreError(
                f"Mismatch: received {len(chunks)} chunks but {num_vectors} embeddings."
            )

        if self.index is None:
            self._init_index(dim)
        elif self.dimension != dim:
            raise VectorStoreError(
                f"Vector dimension mismatch: store has {self.dimension}, incoming is {dim}"
            )

        # Ensure float32 contiguous array
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        try:
            self.index.add(embeddings)
            self.metadata.extend(chunks)
            logger.info(f"Added {num_vectors} vectors to FAISS index. Total: {self.total_vectors}")
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS: {e}")
            raise VectorStoreError(f"Failed to add vectors to FAISS: {e}") from e

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[dict[str, Any], float]]:
        """Search the FAISS vector index for top-k similar chunks.

        Args:
            query_embedding: 1D or 2D query embedding array.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of tuples: (chunk_metadata, similarity_score).
        """
        if self.index is None or self.total_vectors == 0:
            return []

        if query_embedding.ndim == 1:
            query_vec = np.expand_dims(query_embedding, axis=0)
        else:
            query_vec = query_embedding

        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        k = min(top_k, self.total_vectors)
        try:
            scores, indices = self.index.search(query_vec, k)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise VectorStoreError(f"FAISS search failed: {e}") from e

        results: list[tuple[dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk_data = self.metadata[idx]
            results.append((chunk_data, float(score)))

        return results

    def save(self, db_dir: Path | str | None = None) -> None:
        """Persist FAISS index and metadata to disk."""
        target_dir = Path(db_dir or self.db_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        meta_file = target_dir / "metadata.json"

        if self.index is not None:
            try:
                faiss.write_index(self.index, str(index_file))
                with meta_file.open("w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Persisted FAISS index ({self.total_vectors} vectors) to {target_dir}")
            except Exception as e:
                logger.error(f"Failed to save FAISS store to {target_dir}: {e}")
                raise VectorStoreError(f"Failed to save vector store: {e}") from e

    def load(self, db_dir: Path | str | None = None) -> bool:
        """Load persisted FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if index files do not exist.
        """
        target_dir = Path(db_dir or self.db_dir)
        index_file = target_dir / "index.faiss"
        meta_file = target_dir / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            logger.info(f"No existing FAISS index found at {target_dir}")
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d
            with meta_file.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded FAISS index with {self.total_vectors} vectors from {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {target_dir}: {e}")
            raise VectorStoreError(f"Failed to load vector store: {e}") from e

    def clear(self) -> None:
        """Clear in-memory index and metadata."""
        self.index = None
        self.dimension = None
        self.metadata = []
        if self.dimension is not None:
            self._init_index(self.dimension)
