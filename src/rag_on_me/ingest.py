from rag_on_me.data_loader import load_all_documents
from rag_on_me.chunking import split_documents
from rag_on_me.embedding import EmbeddingPipeline
from rag_on_me.vector_store import VectorStore


if __name__ == "__main__":
    docs = load_all_documents("data_dir")
    chunks = split_documents(docs)

    embedder = EmbeddingPipeline()
    embeddings = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.add_documents(chunks, embeddings)
