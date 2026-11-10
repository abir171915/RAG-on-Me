FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data_dir

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles ~2 GB of CUDA libraries.
# Only torch is pinned here: the PyTorch index doesn't host every version of its dependencies.
COPY constraints.txt .
RUN pip install "$(grep '^torch==' constraints.txt)" --index-url https://download.pytorch.org/whl/cpu

# Project and its dependencies, pinned to the versions tested locally
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . -c constraints.txt

# Bake the embedding model into the image so startup needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

# Only the built vector store, not the source documents
COPY data_dir/vector_store ./data_dir/vector_store

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "rag_on_me.api:app", "--host", "0.0.0.0", "--port", "8000"]
