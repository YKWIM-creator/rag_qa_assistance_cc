# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) assistant that answers student questions grounded in documentation scraped from 11 CUNY senior college websites. It uses ChromaDB for vector storage and supports multiple LLM/embedding providers.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env  # Set OPENAI_API_KEY or ANTHROPIC_API_KEY + LLM_PROVIDER
```

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_chain.py -v

# Run a single test
pytest tests/test_chain.py::test_ask_returns_response -v

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit UI (requires API running)
streamlit run ui/app.py

# Scrape a single school (testing)
python scripts/run_scrape.py --school baruch --max-pages 50

# Scrape all 11 CUNY schools
python scripts/run_scrape.py --max-pages 500

# Force re-scrape (clears SQLite queue + pages for the school before crawling)
python scripts/run_scrape.py --school baruch --force-rescrape
```

## Architecture

**Shared data types** live in `src/models.py` — import from here, not from the individual modules:
- `ScrapedPage` — scraper output (url, school, text as Markdown, title, page_type, scraped_at)
- `RewrittenQuery` — query rewriter output (query, school)
- `RAGResponse` — chain output (answer, sources)
- `QuestionRequest` / `AnswerResponse` — FastAPI request/response schemas

Data flows through five sequential stages:

**1. Scraper** (`src/scraper/`)
- `spider.py`: Async BFS crawler with rate limiting (1s delay) and SQLite-backed queue. Single HTTP fetch per URL. Returns `ScrapedPage` objects with `text` set to Markdown.
- `cleaner.py`: Strips nav/footer/scripts, finds main content area, converts to Markdown via `markdownify` (ATX headings, lists preserved). Entry point: `clean_to_markdown(html, url)`.
- `db.py`: `ScraperDB` wraps a SQLite file (`./scraper_cache/scraper.db`) with `queue` and `pages` tables. Enables resume-on-crash and cross-run deduplication via SHA-256 content hashing.
- `filters.py`: `should_skip_url(url)` gates every URL before fetch — skips binary extensions, low-value paths (`/login`, `/calendar`), and noisy query params (`?print=`, `?sort=`).

**2. Ingestion** (`src/ingestion/`)
- `chunker.py`: Splits text using `RecursiveCharacterTextSplitter` (2000 chars, 200 overlap). Attaches metadata: `url`, `school`, `title`, `chunk_index`, `scraped_at`.
- `embedder.py`: Provider abstraction supporting OpenAI (`text-embedding-3-small`), Anthropic/VoyageAI (`voyage-3`), and Ollama.
- `pipeline.py`: Orchestrates scraping → chunking → vector store indexing.

**3. Vector Store** (`src/retrieval/retriever.py`)
- ChromaDB persisted to `./vectorstore/`, collection `cuny_docs`.
- `build_vectorstore()` indexes chunks in batches of 100.
- `get_retriever()` returns an MMR retriever (k=5) to reduce redundancy.

**4. Generation** (`src/generation/`)
- `rewriter.py`: LLM call that extracts the target school and rewrites the question into retrieval-optimized academic language. Returns `RewrittenQuery`.
- `providers.py`: Returns the appropriate LangChain LLM (OpenAI `gpt-4o`, Anthropic `claude-3-5-sonnet-20241022`, or Ollama), all at temperature=0.
- `chain.py`: Core RAG logic — rewrites query → retrieves with optional school metadata filter → formats context labeled `[School | page_type | section]` → invokes LLM via LCEL chain. Falls back to "no information" if retrieval returns nothing. Returns `RAGResponse` with deduplicated source URLs.

**5. API & UI** (`src/api/main.py`, `ui/app.py`)
- FastAPI loads all components at startup via lifespan context. Endpoints: `GET /health`, `POST /ask`, `GET /sources`.
- Streamlit UI calls the API backend and renders answers with source expanders.

## Configuration

`config/settings.py` uses Pydantic `BaseSettings` — all fields are overridable via `.env`. Key settings:
- `LLM_PROVIDER`: `openai` | `anthropic` | `ollama`
- `EMBEDDING_PROVIDER`: `openai` | `anthropic` | `ollama`
- `CHROMA_PERSIST_DIR`: defaults to `./vectorstore`
- CUNY school URLs are hardcoded in `settings.py` as a dict keyed by school short name.

## Evaluation

RAGAS pipeline in `src/evaluation/eval.py` runs against `data/golden_dataset.json` (3 hand-written Q&A pairs). Metrics: `faithfulness`, `answer_relevancy`, `context_recall`. Target thresholds (from design docs): faithfulness ≥ 0.8, answer_relevancy ≥ 0.75.

## Testing Notes

- 81 tests across 13 modules — all should pass (excluding `test_eval.py` which has a pre-existing collection error unrelated to unit logic).
- Tests use `pytest-mock` and `pytest-asyncio`; async tests require `@pytest.mark.asyncio`.
- The spider and API tests mock HTTP calls; no network access required to run tests.
- Spider tests patch `src.scraper.spider.httpx.AsyncClient` (not the top-level `httpx.AsyncClient`).
- `ScraperDB` tests use `tmp_path` fixture for isolated SQLite files per test.
- There is no test coverage for the Streamlit UI.

---

## Figma MCP Integration Rules

This project has **no frontend design system** — the only UI is a Streamlit app (`ui/app.py`). There are no design tokens, CSS frameworks, component libraries, icons, or asset pipelines. Any Figma work applies to the Streamlit UI layer only.

### Design System Analysis

| Aspect | Reality |
|--------|---------|
| UI framework | Streamlit (Python) — not React/Vue |
| Styling | Streamlit built-in theming only; no CSS files |
| Design tokens | None — use `.streamlit/config.toml` for theming (create if absent) |
| Component library | None — uses `st.*` Streamlit primitives |
| Icons | Streamlit emoji strings only (e.g. `"🎓"`) — no icon library |
| Assets | `ui/assets/` if introduced; none currently in repo |
| Build system | None — `streamlit run ui/app.py` executes directly |
| Entry point | `ui/app.py` — single-file Streamlit app |
| Backend API | FastAPI at `http://localhost:8000` — `POST /ask`, `GET /health`, `GET /sources` |
| Answer content | Markdown (from scraper redesign) — render with `st.markdown()`, not `st.write()` |

### Required Figma-to-Code Flow

1. Run `get_design_context` on the target Figma node
2. Run `get_screenshot` for visual reference
3. Translate the design into **Streamlit Python** — not React, not HTML/CSS
4. Map Figma layout/visual concepts to Streamlit equivalents:
   - Columns → `st.columns()`
   - Cards/containers → `st.container()` or `st.expander()`
   - Chat bubbles → `st.chat_message()`
   - Forms → `st.form()` / `st.text_input()` / `st.button()`
   - Color/theme → `.streamlit/config.toml` (`[theme]` section)
   - Markdown answers → always `st.markdown()` (answers contain headings/lists)
5. Validate the rendered UI against the Figma screenshot before marking complete

### Styling Rules

- IMPORTANT: Do not introduce CSS files, Tailwind, or styled-components — this is a Python/Streamlit project
- Custom theming goes in `.streamlit/config.toml` (create if absent); never hardcode hex colors inline
- Responsive layout is handled by Streamlit's built-in column system — do not use CSS grid/flex manually
- IMPORTANT: Do not add any npm/JS dependencies
- Page config (`st.set_page_config`) is already set at the top of `ui/app.py` — do not add a second call

### Asset Rules

- IMPORTANT: If Figma MCP returns a localhost image source, use it directly in `st.image()`
- Static assets (logos, images) belong in `ui/assets/` if introduced
- Do not create a separate asset pipeline
- IMPORTANT: Do not install icon packages — use emoji strings for icons

### Answer Rendering Rule

The RAG chain returns Markdown-formatted answers (headings, lists, bold text). Always render answers with `st.markdown(answer)`, never `st.write(answer)` or `st.text(answer)`.

### If a Frontend Rewrite is Needed

If a task requires a richer UI than Streamlit supports, the right migration target for this project is **FastAPI + a lightweight React frontend** (the backend API already exists). Any such rewrite should:
- Place React components in `ui/src/components/`
- Use the existing `POST /ask` and `GET /sources` API endpoints
- Follow standard React + TypeScript conventions with Tailwind for styling
