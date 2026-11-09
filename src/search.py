from typing import List, Dict, Any
from embedding import EmbeddingPipeline
from vector_store import VectorStore



class Retriever:
    """Find the stored chunks most similar to the asked question."""

    def __init__(self, embedder: EmbeddingPipeline, store: VectorStore):
        self.embedder = embedder
        self.store = store


    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        qeury_embedding = self.embedder.embed_query(query)
        results = self.store.collection.query(
            query_embeddings = [qeury_embedding.tolist()],
            n_results = top_k,
        )

        all_results = []

        for text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            all_results.append({"text": text,
                                "metadata": meta,
                                "similarity": 1-dist})
        
        return all_results
    def search_many(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for query in queries:
            for hit in self.search(query, top_k=top_k):
                key = hit["text"]
                if key not in best or hit["similarity"] > best[key]["similarity"]:
                    best[key] = hit

        ranked = sorted(best.values(), key=lambda h: h["similarity"], reverse=True)
        return ranked[:top_k]

