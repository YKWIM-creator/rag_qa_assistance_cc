# RAG Quality Improvements Design

**Date:** 2026-03-30
**Status:** Approved

## Goals

Address three RAG quality problems:
1. **Irrelevant answers** — chunks don't match the question well
2. **Incomplete answers** — LLM has right docs but missing section context
3. **Wrong school** — retriever pulls chunks from the wrong CUNY college

## Approach: Full-Stack (Scraper + Chunker + Retriever + Prompt)

### Prerequisites
- Built on top of the scraper redesign (`feat/scraper-redesign` branch): SQLite queue, markdownify output, URL filters, content dedup.
- Requires re-scrape + re-index after changes.

---

## Section 1: Scraper — Sitemap Seeding + Page Type Classification

### Sitemap seeding (`spider.py` + `db.py`)

Before BFS from the homepage, the crawler fetches `<school_url>/sitemap.xml`. All URLs found are bulk-inserted into the SQLite queue as `pending`. BFS then continues from those seeds.

- Fetch `sitemap.xml` at crawl start; also check `robots.txt` for `Sitemap:` directive
- Parse all `<loc>` entries; `enqueue_new()` them as `pending`
- Fall back to homepage-only BFS if sitemap not found (HTTP 404)
- Apply existing `should_skip_url()` filter to sitemap URLs too

### Page type classification (new `src/scraper/classifier.py`)

Classify each scraped page into one of 5 types at scrape time, based on URL path patterns and `<h1>` text:

| Type | URL/heading signals |
|---|---|
| `admissions` | `/admissions`, `/apply`, `/requirements`, `/enrollment` |
| `academics` | `/academics`, `/programs`, `/majors`, `/departments`, `/courses` |
| `financial_aid` | `/financial`, `/aid`, `/scholarships`, `/tuition`, `/fees` |
| `student_services` | `/housing`, `/advising`, `/health`, `/career`, `/clubs` |
| `general` | everything else |

- `classify_page(url, h1_text) -> str` — pure function, easy to test
- `page_type` stored as new column in `ScraperDB.pages`
- Passed as field on `ScrapedPage` dataclass

---

## Section 2: Chunker — Markdown-Aware Splitting + Enriched Metadata

### Two-pass chunking pipeline (`chunker.py`)

Replace `RecursiveCharacterTextSplitter` alone with a two-pass approach:

1. **Pass 1:** `MarkdownHeaderTextSplitter` splits on `#`, `##`, `###` — keeps each section together
2. **Pass 2:** `RecursiveCharacterTextSplitter` (1500 chars, 150 overlap) handles sections that are still too large after header splitting

### Enriched chunk metadata

```python
{
    "url": "https://www.baruch.cuny.edu/admissions",
    "school": "baruch",
    "title": "Baruch Admissions | CUNY",
    "section_heading": "Undergraduate Requirements",  # from MarkdownHeaderTextSplitter
    "page_type": "admissions",                        # from scraper classifier
    "chunk_index": 2,
    "scraped_at": "2026-03-30T..."
}
```

- `section_heading`: nearest `##`/`###` header above the chunk; falls back to page title if none
- `page_type`: passed through from `ScrapedPage`

---

## Section 3: Retriever — Query Rewriting + School-Aware Filtering

### Query rewriting (new `src/generation/rewriter.py`)

Before hitting the vector store, the question passes through a lightweight LLM call:

**Input:** raw user question
**Output:** `{ "school": "baruch" | null, "query": "<rewritten question>" }`

Example:
```
Input:  "what are the requirements to get into baruch?"
Output: { "school": "baruch", "query": "Baruch College undergraduate admissions requirements" }
```

- If `school` is detected → apply `where={"school": school}` metadata filter to ChromaDB
- If `school` is `null` → search all schools normally
- Rewritten query replaces the original for embedding lookup

### Retriever tuning (`retriever.py`)

- `fetch_k`: 15 → 25 (more MMR candidates)
- `k`: stays at 5
- Optional secondary filter: if question contains financial aid keywords, prefer `page_type=financial_aid` chunks

---

## Section 4: Prompt — Section-Aware Context Formatting

### Richer context format (`chain.py`)

Each chunk in the context block is prefixed with provenance:

```
[Baruch College | admissions | Undergraduate Requirements]
To apply to Baruch, you must submit...

---

[Hunter College | admissions | Transfer Admissions]
Transfer students must have completed...
```

### Updated prompt template

```
You are a CUNY student assistant. Answer the question using ONLY the context below.
Each context block is labeled with [School | page type | section].
If the question asks about a specific school, prioritize blocks from that school.
If the answer is not found in the context, say:
"I don't have information about that in the CUNY documents I've indexed."
Be concise and cite the school name in your answer.

Context:
{context}

Question: {question}

Answer:
```

---

## Files Changed

| File | Change |
|---|---|
| `src/scraper/spider.py` | Add sitemap fetch + seed before BFS |
| `src/scraper/db.py` | Add `page_type` column to `pages` table |
| `src/scraper/classifier.py` | New — `classify_page(url, h1_text) -> str` |
| `src/scraper/spider.py` | Pass `page_type` on `ScrapedPage` |
| `src/ingestion/chunker.py` | Two-pass Markdown-aware splitting + enriched metadata |
| `src/retrieval/retriever.py` | Raise `fetch_k` to 25; support metadata filter param |
| `src/generation/rewriter.py` | New — LLM-based school extraction + query rewriting |
| `src/generation/chain.py` | Use rewriter; richer `_format_docs`; updated prompt |

## Out of Scope

- Changes to Streamlit UI
- Changes to FastAPI endpoints
- RAGAS evaluation update (separate task)
- Parallel school crawling
