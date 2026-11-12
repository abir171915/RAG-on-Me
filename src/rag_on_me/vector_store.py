import chromadb
import numpy as np
from typing import Any, Dict, List
from langchain_core.documents import Document


class VectorStore:
    """Persist chunk embeddings, text and metadata in ChromaDB."""

    def __init__(self, persist_dir: str = "data_dir/vector_store", collection_name: str = "abir_profile"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_by_source_type(self, source_type: str) -> List[Dict[str, Any]]:
        """Return every chunk of one source type (e.g. "profile"), in document order."""
        result = self.collection.get(where={"source_type": source_type}, include=["documents", "metadatas"])
        chunks = [{"id": i, "text": t, "metadata": m}
                  for i, t, m in zip(result["ids"], result["documents"], result["metadatas"])]

        def document_order(chunk):                     # IDs look like "aboutme.md::0"
            source, _, index = chunk["id"].rpartition("::")
            return source, int(index)

        return sorted(chunks, key=document_order)

    def add_documents(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        # Number chunks within each source, so adding or removing one document
        # never shifts the IDs of chunks from other documents
        counts: Dict[str, int] = {}
        ids = []
        for c in chunks:
            source = c.metadata.get("source_name") or c.metadata.get("source", "unknown")
            ids.append(f"{source}::{counts.get(source, 0)}")
            counts[source] = counts.get(source, 0) + 1

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        print(f"[INFO] Stored {len(chunks)} chunks. Collection size: {self.collection.count()}")
