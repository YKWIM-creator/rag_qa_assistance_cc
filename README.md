# CUNY RAG Assistant

A retrieval-augmented generation (RAG) system that answers student questions grounded in scraped CUNY college documentation.

## Overview

The assistant scrapes public pages from CUNY senior college websites, indexes them into a vector store, and uses an LLM to generate answers that cite real source documents. It exposes a REST API and a chat UI.

**Tech stack:** Python 3.11+, LangChain, ChromaDB, httpx, BeautifulSoup4, FastAPI, Streamlit, RAGAS

## Project Structure

```
config/          # Settings via pydantic-settings
src/
  scraper/       # Async BFS web crawler + HTML cleaner + page classifier
  ingestion/     # Text chunker, embedding provider, ingestion pipeline
  retrieval/     # ChromaDB vector store + MMR retriever
  generation/    # LLM provider abstraction + query rewriter + RAG chain
  api/           # FastAPI backend
  evaluation/    # RAGAS evaluation pipeline + dataset generator + report writer
scripts/         # run_scrape.py — CLI to trigger scraping
ui/              # Streamlit chat UI
tests/           # Unit tests (51 tests, all passing)
data/            # golden_dataset.json for evaluation
eval_reports/    # Auto-generated Markdown evaluation reports
```

## Quickstart

### 1. Install dependencies

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (or ANTHROPIC_API_KEY for Anthropic)
```

Key settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `LLM_PROVIDER` | `openai` | `openai` / `anthropic` / `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./vectorstore` | Where ChromaDB stores data |

### 3. Scrape and index

```bash
# Scrape one school (fast, good for testing)
python scripts/run_scrape.py --school baruch --max-pages 50

# Scrape all 11 CUNY senior colleges
python scripts/run_scrape.py --max-pages 500

# Force re-scrape (clears cached pages for the school before crawling)
python scripts/run_scrape.py --school baruch --force-rescrape
```

Supported schools: `baruch`, `brooklyn`, `city`, `hunter`, `john_jay`, `lehman`, `medgar_evers`, `nycct`, `queens`, `staten_island`, `york`

### 4. Start the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What programs does Baruch College offer?"}'
```

### 5. Start the chat UI (optional)

```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/ask` | Answer a question (`{"question": "..."}`) |
| `GET` | `/sources` | List indexed schools |

`/ask` response:
```json
{
  "answer": "Baruch College offers programs in...",
  "sources": [
    {"url": "https://www.baruch.cuny.edu/...", "school": "baruch", "title": "Programs"}
  ]
}
```

## RAG Pipeline

Each question flows through five steps:

1. **Query rewriting** (`src/generation/rewriter.py`) — The LLM extracts the target school (if any) and rewrites the question into retrieval-optimized academic language.
2. **Retrieval** (`src/retrieval/retriever.py`) — MMR retriever (k=5) queries ChromaDB, optionally scoped to the extracted school via a metadata filter.
3. **Context formatting** (`src/generation/chain.py`) — Retrieved chunks are labeled `[School | page_type | section]` so the LLM knows which college each block comes from.
4. **Generation** (`src/generation/chain.py`) — An LCEL chain formats the prompt and invokes the LLM. Falls back to a "no information" response if retrieval returns nothing.
5. **Source deduplication** — Returned source URLs are deduplicated before the `RAGResponse` is returned.

The scraper also classifies each page (`src/scraper/classifier.py`) into one of five content types (`admissions`, `academics`, `financial_aid`, `student_services`, `general`) based on URL path and h1 text, attaching the result as `page_type` metadata on each chunk.

## Running Tests

```bash
pytest tests/ -v
# 51 passed
```

## Evaluation

### Running RAGAS metrics

```bash
python -c "
from src.evaluation.eval import run_evaluation
from src.ingestion.embedder import get_embedding_model
from src.retrieval.retriever import load_vectorstore, get_retriever
from src.generation.providers import get_llm

embeddings = get_embedding_model()
vs = load_vectorstore(embeddings)
retriever = get_retriever(vs)
llm = get_llm()
results = run_evaluation(retriever, llm)
print(results)
"
```

Each run is persisted to SQLite and compared against the previous run:

```
Run #3  (commit: 657a22c)
┌─────────────────────┬────────┬────────┬──────────┐
│ Metric              │  Prev  │  Now   │  Δ       │
├─────────────────────┼────────┼────────┼──────────┤
│ faithfulness        │   0.82 │   0.85 │ +0.0300↑ │
│ answer_relevancy    │   0.78 │   0.80 │ +0.0200↑ │
│ context_recall      │   0.71 │   0.74 │ +0.0300↑ │
│ context_precision   │   0.69 │   0.72 │ +0.0300↑ │
│ answer_correctness  │   0.76 │   0.79 │ +0.0300↑ │
└─────────────────────┴────────┴────────┴──────────┘
```

A Markdown report is written to `eval_reports/` after each run.

Target thresholds: faithfulness ≥ 0.8, answer_relevancy ≥ 0.75.

### Building the golden dataset

Use the semi-automated dataset generator to generate and review QA pairs from scraped pages:

```bash
# Generate QA candidates from raw markdown files for a school
python -m src.evaluation.dataset_generator --generate --school baruch

# Interactively approve, edit, or skip candidates → appends to golden_dataset.json
python -m src.evaluation.dataset_generator --review --school baruch
```

## LLM Providers

Switch providers by setting `LLM_PROVIDER` and `EMBEDDING_PROVIDER` in `.env`:

| Provider | LLM | Embeddings |
|---|---|---|
| `openai` | `gpt-4o` | `text-embedding-3-small` |
| `anthropic` | `claude-3-5-sonnet-20241022` | `voyage-3` (via VoyageAI) |
| `ollama` | any local model | any local model |
