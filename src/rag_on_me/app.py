from rag_on_me.config import settings
from rag_on_me.embedding import EmbeddingPipeline
from rag_on_me.vector_store import VectorStore
from rag_on_me.search import Retriever
from rag_on_me.generator import RagGenerator
from rag_on_me.query_rewriter import QueryRewriter


if __name__ == "__main__":
    store = VectorStore(str(settings.vector_store_dir), settings.collection_name)
    if store.collection.count() == 0:
        raise SystemExit("Vector store is empty. Run: python -m rag_on_me.ingest")

    embedder = EmbeddingPipeline(settings.embedding_model)
    rewriter = QueryRewriter(settings.rewriter_model, api_key=settings.groq_api_key)
    rag = RagGenerator(Retriever(embedder, store), rewriter,
                       model_name=settings.llm_model, api_key=settings.groq_api_key,
                       pinned_chunks=store.get_by_source_type("profile"))

    while True:
        question = input("\nAsk about Abir (or 'q' to quit): ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            print(rag.answer(question, top_k=settings.top_k)["answer"])
