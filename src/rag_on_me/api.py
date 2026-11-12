"""HTTP API for the RAG system. Run with: uvicorn rag_on_me.api:app"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_on_me.config import settings
from rag_on_me.embedding import EmbeddingPipeline
from rag_on_me.vector_store import VectorStore
from rag_on_me.search import Retriever
from rag_on_me.query_rewriter import QueryRewriter
from rag_on_me.generator import RagGenerator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and connect to the store once, before the server accepts requests."""
    store = VectorStore(str(settings.vector_store_dir), settings.collection_name)
    if store.collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run: python -m rag_on_me.ingest")

    embedder = EmbeddingPipeline(settings.embedding_model)
    rewriter = QueryRewriter(settings.rewriter_model, api_key=settings.groq_api_key)
    app.state.store = store
    app.state.rag = RagGenerator(Retriever(embedder, store), rewriter,
                                 model_name=settings.llm_model, api_key=settings.groq_api_key,
                                 pinned_chunks=store.get_by_source_type("profile"))
    yield


app = FastAPI(title="Ask Abir", lifespan=lifespan)

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "chunks": request.app.state.store.collection.count()}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    try:
        result = request.app.state.rag.answer(body.question, top_k=settings.top_k)
    except Exception as e:
        print(f"[ERROR] /ask failed: {e!r}")
        raise HTTPException(status_code=502, detail="Answer generation failed. Please try again.")
    return AskResponse(**result)
