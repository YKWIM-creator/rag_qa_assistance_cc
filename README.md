# CUNY RAG Assistant

A retrieval-augmented generation (RAG) system that answers student questions grounded in scraped CUNY college documentation.

## Overview

The assistant scrapes public pages from CUNY senior college websites, indexes them into a vector store, and uses an LLM to generate answers that cite real source documents. It exposes a REST API and a chat UI.

**Tech stack:** Python 3.11+, LangChain, ChromaDB, httpx, BeautifulSoup4, FastAPI, Streamlit, RAGAS

## Project Structure

```
config/          # Settings via pydantic-settings
src/
  scraper/       # Async BFS web crawler + HTML cleaner
  ingestion/     # Text chunker, embedding provider, ingestion pipeline
  retrieval/     # ChromaDB vector store + MMR retriever
  generation/    # LLM provider abstraction + RAG chain
  api/           # FastAPI backend
  evaluation/    # RAGAS evaluation pipeline
scripts/         # run_scrape.py — CLI to trigger scraping
ui/              # Streamlit chat UI
tests/           # Unit tests (28 tests, all passing)
data/            # golden_dataset.json for evaluation
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

## Running Tests

```bash
pytest tests/ -v
# 28 passed
```

## Evaluation

Run RAGAS metrics (faithfulness, answer relevancy, context recall) against the golden dataset:

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

## LLM Providers

Switch providers by setting `LLM_PROVIDER` and `EMBEDDING_PROVIDER` in `.env`:

| Provider | LLM | Embeddings |
|---|---|---|
| `openai` | `gpt-4o` | `text-embedding-3-small` |
| `anthropic` | `claude-3-5-sonnet-20241022` | `voyage-3` (via VoyageAI) |
| `ollama` | any local model | any local model |
