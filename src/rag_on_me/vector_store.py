import chromadb
import numpy as np
from typing import List
from langchain_core.documents import Document


class VectorStore:
    """Persist chunk embeddings, text and metadata in ChromaDB."""

    def __init__(self, persist_dir: str = "data_dir/vector_store", collection_name: str = "abir_profile"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        ids = [f"{c.metadata.get('source', 'unknown')}::{i}" for i, c in enumerate(chunks)]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        print(f"[INFO] Stored {len(chunks)} chunks. Collection size: {self.collection.count()}")
