from rag_on_me.config import settings
from rag_on_me.data_loader import load_all_documents
from rag_on_me.chunking import split_documents
from rag_on_me.embedding import EmbeddingPipeline
from rag_on_me.vector_store import VectorStore


if __name__ == "__main__":
    docs = load_all_documents(settings.data_dir)
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)

    embedder = EmbeddingPipeline(settings.embedding_model)
    embeddings = embedder.embed_chunks(chunks)

    store = VectorStore(str(settings.vector_store_dir), settings.collection_name)
    store.add_documents(chunks, embeddings)
