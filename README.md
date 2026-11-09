# Ask Abir: A RAG System About My Professional Background

A retrieval-augmented generation (RAG) system that answers questions about my work: my CV, my MSc thesis, my published research paper, and my GitHub projects. Ask it *"Has Abir worked on medical imaging?"* and it retrieves the relevant passages from my documents and answers using only those sources, citing where each fact came from.

I built it from scratch as a modular pipeline so that I could understand, measure and improve each stage on its own, instead of wiring together a one-file demo.

---

## How it works

```
                    ┌──────────────── INGEST (run once, when data changes) ────────────────┐
                    │                                                                       │
 data_dir/          │  load PDFs + GitHub READMEs  →  chunk  →  embed  →  store in ChromaDB │
 ├─ pdf_file/       │  (PyMuPDF, GitLoader)          (1000c)   (MiniLM)   (persisted)       │
 └─ github/repos.txt│                                                                       │
                    └───────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────── QUERY (every question) ───────────────────────┐
 "Has Abir worked   │  rewrite question  →  retrieve top 5  →  LLM answers from context     │
  on medical        │  (Groq LLM)           (cosine search)     (grounded prompt, cites     │
  imaging?"         │                                            sources, says "I don't     │
                    │                                            know" when unsupported)    │
                    └───────────────────────────────────────────────────────────────────────┘
```

| Stage | Implementation | File |
|---|---|---|
| Load | PDFs via `PyMuPDFLoader` (one Document per page); GitHub repos via `GitLoader` (README + requirements only) | `data_loader.py`, `loading_git.py` |
| Chunk | `RecursiveCharacterTextSplitter`, 1000 chars, 200 overlap | `chunking.py` |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally) | `embedding.py` |
| Store | ChromaDB, persisted to disk, cosine distance, deterministic IDs + upsert | `vector_store.py` |
| Rewrite | LLM rewrites the question into a more specific search query | `query_rewriter.py` |
| Retrieve | Embed query, top-k similarity search, merge and deduplicate | `search.py` |
| Generate | Groq `openai/gpt-oss-20b`, temperature 0, grounded prompt | `generator.py` |
| Evaluate | Hit rate and recall@5 on a labelled question set | `evaluate.py`, `eval_set.py` |

---

## Results

Retrieval is evaluated on 10 labelled questions. Each question lists the source(s) that *should* be retrieved. They include specific facts, skills, paraphrased questions and list questions.

| Configuration | Hit rate@5 | Mean recall@5 |
|---|---|---|
| Baseline (original question only) | 0.80 | 0.72 |
| Multi-query rewriting with a fixed vocabulary list | 0.80 | 0.63 |
| **Single anchored query rewrite (current)** | **1.00** | **0.85** |

- **Hit rate**: the share of questions where at least one correct source appears in the top 5.
- **Recall**: the fraction of all expected sources that were retrieved. This is what matters for list questions like *"what deep learning projects…"*.

---

## Design decisions and what I learned

**Local embeddings, hosted LLM.** The documents include my CV and unpublished thesis work. Embedding them locally keeps that text on my machine, and only the few retrieved chunks are sent to the LLM. MiniLM is also free and fast for a corpus of this size.

**Only READMEs and requirements from GitHub, not code.** Questions like *"what did Abir build?"* are answered by prose, not by source code, which embeds poorly against natural-language questions. `requirements.txt` is included because it reliably lists the tech stack.

**Ingest and query are separate scripts.** Embedding all chunks again for every question was wasteful. `ingest.py` builds the store once, and `app.py` only embeds the incoming question.

**Deterministic chunk IDs with `upsert`.** With random IDs, every re-ingest would duplicate the whole store. Stable IDs make re-running safe.

**Query rewriting: the first version made results worse, and the evaluation showed it.** My first rewriter produced three queries per question, steered by a list of technical terms from my documents. On a single test question it looked like a clear win. On the full evaluation set it *lowered* recall from 0.72 to 0.63: it fixed one question and broke three. Looking at the queries showed why. The rewriter replaced the user's own words with terms from the list, so *"What was the MSc thesis about?"* became *"CNN ResNet image classification"*. That query matched a different project very strongly and pushed the thesis out of the results.

The current rewriter **anchors on the user's key terms** and only adds closely related terms around them. That one change took recall to 0.85.

**Raw similarity scores from different queries are not comparable.** Merging results from several queries by raw cosine similarity lets a confidently *wrong* query dominate. This is why the current design uses a single rewritten query, and why rank-based fusion (RRF) is on the roadmap before multi-query retrieval comes back.

**Fail loudly, not silently.** Several bugs during development were invisible: an encoding issue that made a loader skip a file, an LLM that returned empty output and triggered a silent fallback, a typo in an expected source name. The code now logs warnings when fallbacks fire, and the evaluation refuses to run if an expected source doesn't exist in the store.

---

## Project structure

```
RAG/
├── data_dir/
│   ├── github/repos.txt      # GitHub repos to index, one URL per line: <url> [branch]
│   ├── pdf_file/             # CV and papers (not committed, see below)
│   ├── repo_cache/           # local clones made by GitLoader (generated)
│   └── vector_store/         # ChromaDB files (generated)
├── notebook/                 # exploration notebooks
├── src/
│   ├── ingest.py             # entry point: build the vector store
│   ├── app.py                # entry point: interactive Q&A
│   ├── evaluate.py           # entry point: retrieval evaluation
│   ├── eval_set.py           # labelled evaluation questions
│   ├── data_loader.py        # load PDFs and GitHub docs
│   ├── loading_git.py        # clone repos, load README.md / requirements.txt
│   ├── chunking.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── search.py
│   ├── query_rewriter.py
│   └── generator.py
├── requirements.txt
└── .env                      # GROQ_API_KEY (not committed)
```

---

## Running it locally

**Requirements:** Python 3.12, a free [Groq API key](https://console.groq.com).

```powershell
# 1. Create and activate a virtual environment
python -m venv myrag
.\myrag\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key to a .env file in the project root
Set-Content -Path .env -Value "GROQ_API_KEY=your_key_here" -Encoding ascii
```

(Don't use `echo ... > .env` in Windows PowerShell 5.1. It writes the file as UTF-16, which `python-dotenv` can't read.)

**Add your own documents.** The personal PDFs are not included in this repository.
- Put PDFs in `data_dir/pdf_file/`
- List GitHub repos in `data_dir/github/repos.txt`, one URL per line (an optional branch after the URL; the default is `main`)

Run all commands from the project root:

```powershell
python src\ingest.py      # build the vector store (first run clones the repos)
python src\app.py         # ask questions; type 'q' to quit
python src\evaluate.py    # measure retrieval quality
```

Re-run `ingest.py` whenever the documents change. If you **remove** a document, delete `data_dir/vector_store/` first: upsert adds and updates entries but never deletes them.

---

## Known limitations

- **List questions are the weak spot.** Questions whose answer spans several projects (e.g. *"medical imaging work"*, recall 0.33) often retrieve several chunks from one project instead of one from each.
- **The data is imbalanced.** The MSc thesis and the research paper make up about 65% of all chunks, so they get more chances to match any question.
- **About 19% of chunks exceed MiniLM's 256-token limit** and are truncated during embedding.
- **The evaluation set is small** (10 questions, single run), so differences of one question can move the scores noticeably.
- **Only retrieval is evaluated.** Answer quality (faithfulness, correct "I don't know") is not yet measured.

## Roadmap

- [ ] Limit chunks per source, and merge multi-query results with Reciprocal Rank Fusion, to improve list questions
- [ ] Hybrid search (BM25 + vectors) for exact technical terms
- [ ] Smaller chunks, or an embedding model with a longer context, to remove truncation
- [ ] Evaluate generated answers, not just retrieval
- [ ] Packaging (`pyproject.toml`), config and logging
- [ ] FastAPI endpoint, UI, Docker and deployment

## Tech stack

Python · LangChain · ChromaDB · sentence-transformers · Groq (gpt-oss-20b) · PyMuPDF · GitPython
