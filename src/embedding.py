import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from data_loader import load_all_documents
from chunking import split_documents

class EmbeddingPipeline:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)


    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks.... (IN PROGRESS)")
        embeddings = self.model.encode(texts, show_progress_bar = True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        
        return embeddings