# CUNY RAG Assistant — Design Document

**Date:** 2026-03-27
**Approach:** LangChain-Centered RAG Pipeline

---

## 1. Overview

A Retrieval-Augmented Generation (RAG) assistant that answers student questions grounded in real CUNY documentation. The user asks a question, the system retrieves the most relevant chunks from indexed CUNY web content, and an LLM generates an answer grounded in those chunks.

**Pipeline:** Data Collection → Chunking & Embedding → Retrieval → LLM Generation → Evaluation

---

## 2. Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Orchestration | LangChain | Mature ecosystem, built-in provider abstractions, RAG tooling |
| Vector Store | ChromaDB | Local, zero setup, persistent, great for prototyping |
| LLM/Embeddings | Abstraction layer | Support OpenAI, Anthropic, and Ollama interchangeably |
| Data Source | Web scraping | All 11 CUNY senior college websites, broad crawl |
| Evaluation | RAGAS | Full pipeline eval: faithfulness + answer relevancy + context recall |
| Backend | FastAPI | REST API: /ask, /health, /sources |
| Frontend | Streamlit | Chat interface with source citations |

---

## 3. Architecture

```
cuny.edu (11 senior colleges)
        │
        ▼
┌─────────────────┐
│  Data Collection │  httpx + BeautifulSoup (async crawl, broad scope)
│  (Web Scraper)   │  → raw HTML → cleaned text → saved to disk
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chunking &       │  LangChain RecursiveCharacterTextSplitter
│ Embedding        │  → LangChain embedding abstraction (OpenAI / Anthropic / local)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB       │  Persistent local vector store
│  (Vector Store)  │  → chunks + metadata (school, URL, page_title, scraped_at)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval      │  LangChain retriever → top-k similarity + MMR reranking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation  │  LangChain LLM abstraction → GPT-4o / Claude / Ollama
│  (RAG Chain)     │  → prompt template with retrieved context + question
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI        │  /ask, /health, /sources
│   Backend        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI    │  Chat interface + source citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation     │  RAGAS: faithfulness, answer relevancy, context recall
└─────────────────┘
```

---

## 4. Project Structure

```
cuny-rag/
├── data/
│   ├── raw/              # scraped HTML/text per school
│   └── processed/        # cleaned, chunked docs
├── vectorstore/          # ChromaDB persistent storage
├── src/
│   ├── scraper/
│   │   ├── spider.py     # async crawler (httpx + BeautifulSoup)
│   │   └── cleaner.py    # HTML → clean text
│   ├── ingestion/
│   │   ├── chunker.py    # LangChain text splitters
│   │   └── embedder.py   # LangChain embedding abstraction
│   ├── retrieval/
│   │   └── retriever.py  # ChromaDB + LangChain retriever
│   ├── generation/
│   │   ├── chain.py      # RAG chain (prompt + LLM)
│   │   └── providers.py  # OpenAI / Anthropic / Ollama config
│   ├── api/
│   │   └── main.py       # FastAPI app
│   └── evaluation/
│       └── eval.py       # RAGAS metrics runner
├── ui/
│   └── app.py            # Streamlit chat interface
├── config/
│   └── settings.py       # env vars, model config, school URLs
├── tests/
├── .env
└── requirements.txt
```

---

## 5. Data Flow

### Stage 1 — Data Collection
Async crawler visits all 11 CUNY senior college domains. Pages are cleaned (strip nav/footer/ads, extract main content) and saved as plain text with metadata: `{school, url, page_title, scraped_at}`. Respects `robots.txt`, rate-limited to avoid bans.

**Target schools:** Baruch, Brooklyn, City College, Hunter, John Jay, Lehman, Medgar Evers, NYC College of Technology, Queens, Staten Island, York.

### Stage 2 — Chunking & Embedding
LangChain `RecursiveCharacterTextSplitter` splits docs into ~500 token chunks with 50-token overlap. Each chunk is embedded via the configured provider and stored in ChromaDB with its metadata.

### Stage 3 — Retrieval
Query is embedded with the same model. ChromaDB returns top-k (default: 5) most similar chunks. MMR (max marginal relevance) reduces redundancy across chunks from the same page.

### Stage 4 — LLM Generation
Prompt template:
```
You are a CUNY student assistant. Answer using ONLY the context below.
If the answer isn't in the context, say so.

Context: {chunks}
Question: {question}
Answer:
```

### Stage 5 — Evaluation
RAGAS scores each Q&A pair on faithfulness, answer relevancy, and context recall.

---

## 6. Error Handling

**Scraper**
- Skip non-200 pages, log for retry
- 10s timeout per request, 3 retries with exponential backoff
- Dedup via seen-set (keyed by URL)

**Ingestion**
- Validate chunks are non-empty before embedding
- Retry failed embedding calls with backoff; log failed chunks for re-run
- ChromaDB writes are idempotent (keyed by URL + chunk index)

**Retrieval & Generation**
- 0 results → return graceful "no information found", skip LLM call
- LLM API errors → 503 with user-friendly message
- All errors logged with full context

**API**
- Pydantic input validation (question length limits, sanitization)
- Global exception handler returns structured JSON errors

---

## 7. Testing Strategy

**Unit Tests**
- Chunker: chunk sizes within token limits, overlap correct
- Cleaner: HTML stripping preserves main content
- Prompt builder: context + question correctly interpolated
- Provider config: correct model selected per env setting

**Integration Tests**
- Ingestion pipeline: scrape test page → chunk → embed → verify in ChromaDB
- Retrieval: known question returns expected chunks
- RAG chain: end-to-end with mock LLM

**Evaluation (RAGAS)**
- Golden dataset: 20-30 hand-written Q&A pairs per school
- Pass thresholds: faithfulness ≥ 0.8, answer relevancy ≥ 0.75
- Run after any major pipeline change

**Manual Spot-Checking**
- After each school is ingested, run 5-10 real questions and review answers

---

## 8. Scope Boundaries

- **In scope:** End-to-end RAG pipeline, all 11 CUNY senior colleges, FastAPI backend, Streamlit UI, RAGAS evaluation
- **Out of scope:** Production deployment/hosting, user authentication, conversation history/memory, real-time data updates
