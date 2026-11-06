from dotenv import load_dotenv
from embedding import EmbeddingPipeline
from vector_store import VectorStore
from search import Retriever
from generator import RagGenerator

load_dotenv()


if __name__ == "__main__":
    store = VectorStore()
    if store.collection.count() == 0:
        raise SystemExit("Vector store is empty. Run: python src\\ingest.py")

    embedder = EmbeddingPipeline()
    rag = RagGenerator(Retriever(embedder, store))

    while True:
        question = input("\nAsk about Abir (or 'q' to quit): ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            print(rag.answer(question))
