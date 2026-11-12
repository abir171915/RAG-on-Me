FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data_dir

RUN useradd --create-home appuser
WORKDIR /app

# Steps are ordered from least to most frequently changed, so a code edit
# only re-runs the last two steps instead of reinstalling everything.

# CPU-only PyTorch first: the default Linux wheel bundles ~2 GB of CUDA libraries.
# Only torch is pinned here: the PyTorch index doesn't host every version of its dependencies.
COPY constraints.txt .
RUN pip install "$(grep '^torch==' constraints.txt)" --index-url https://download.pytorch.org/whl/cpu

# Dependencies only. A stub package lets pip read pyproject.toml without the real code,
# so this layer is only rebuilt when the dependencies change.
COPY pyproject.toml README.md ./
RUN mkdir -p src/rag_on_me && touch src/rag_on_me/__init__.py \
    && pip install . -c constraints.txt \
    && pip uninstall -y rag-on-me

# Bake the embedding model into the image so startup needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chown -R appuser /app/.cache
ENV HF_HUB_OFFLINE=1

# Only the built vector store, not the source documents
COPY --chown=appuser data_dir/vector_store ./data_dir/vector_store

# Code last
COPY src ./src
RUN pip install --no-deps .

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "rag_on_me.api:app", "--host", "0.0.0.0", "--port", "8000"]
