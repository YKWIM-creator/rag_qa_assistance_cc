# Scraper Redesign Design

**Date:** 2026-03-29
**Status:** Approved

## Goals

1. **Reliability** — persist crawl state to SQLite so crashes resume where they left off; no pages lost mid-run.
2. **Content quality** — markdownify output preserves heading/list structure; URL filtering and content dedup eliminate low-value pages.
3. **Correctness** — fix double-fetch bug (current code fetches each page twice).

## Architecture

### New files

| File | Purpose |
|---|---|
| `src/scraper/db.py` | `ScraperDB` class — wraps SQLite, manages `pages` and `queue` tables |
| `src/scraper/filters.py` | `should_skip_url(url)` — pure filter function, no side effects |

### Modified files

| File | Change |
|---|---|
| `src/scraper/spider.py` | Single-fetch crawl loop, SQLite-backed queue, markdownify output |
| `src/scraper/cleaner.py` | Replace `get_text()` with `markdownify()` pipeline |
| `scripts/run_scrape.py` | Add `--force-rescrape` flag |

---

## Section 1: SQLite Schema

Database path: `./scraper_cache/scraper.db` (configurable via `SCRAPER_DB_PATH` env var).

### `pages` table

```sql
CREATE TABLE pages (
    url          TEXT PRIMARY KEY,
    school       TEXT NOT NULL,
    title        TEXT,
    markdown     TEXT,
    content_hash TEXT,
    scraped_at   TEXT
);
```

### `queue` table

```sql
CREATE TABLE queue (
    url      TEXT PRIMARY KEY,
    school   TEXT NOT NULL,
    status   TEXT NOT NULL DEFAULT 'pending',  -- pending | scraped | failed | skipped
    added_at TEXT
);
```

**Resume behavior:** On re-run, the crawler loads all `pending` rows from `queue` as its starting frontier, skipping anything already `scraped`, `failed`, or `skipped`.

---

## Section 2: URL Filtering (`filters.py`)

`should_skip_url(url: str) -> bool` — returns `True` if the URL should be skipped before any HTTP request.

Rules:
- **Extension filter:** `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.zip`, `.mp4`, `.mp3`
- **Path pattern filter:** `/login`, `/logout`, `/calendar`, `/events`, `/print`, `/feed`, `/wp-json`, `?page=`, `?sort=`, `?filter=`
- **DB check:** URL already present in `queue` with status `scraped` or `skipped`

---

## Section 3: Redesigned Crawl Loop (`spider.py`)

**Key fix — single fetch per URL:**

Current code fetches each URL twice (once in `crawl_school`, once in `scrape_page`). The redesign passes `response.text` directly to the parser, eliminating the duplicate request.

**New `crawl_school` flow:**

```
1. seed_queue(school, start_url) → insert start_url as pending if not seen
2. while pending URLs exist and len(results) < max_pages:
     url = db.next_pending()
     if should_skip_url(url): db.mark(url, 'skipped'); continue
     response = await client.get(url)
     if not 200: db.mark(url, 'failed'); continue
     markdown = clean_to_markdown(response.text, url)
     if not markdown: db.mark(url, 'skipped'); continue
     content_hash = sha256(markdown)
     if db.hash_exists(content_hash): db.mark(url, 'skipped'); continue  # dedup
     db.save_page(url, school, title, markdown, content_hash)
     db.mark(url, 'scraped')
     new_links = extract_links(response.text, url)
     db.enqueue_new(new_links, school)
3. return pages from db for this school
```

**`ScrapedPage` dataclass update:**

```python
@dataclass
class ScrapedPage:
    url: str
    school: str
    text: str        # set to markdown value (backward compat with ingestion pipeline)
    title: str = ""
    scraped_at: str = ...
```

---

## Section 4: Markdownify Cleaner (`cleaner.py`)

**New pipeline:**

1. BeautifulSoup strips `REMOVE_TAGS` (nav, footer, header, script, style, aside, iframe, form)
2. Find main content: `main` → `[role=main]` → `article` → `#content` → `.content` → `body`
3. Pass main element's HTML string to `markdownify(main_html, heading_style="ATX")`
4. Return markdown string (empty string if nothing found)

**Why markdownify over `get_text()`:** Preserves heading hierarchy (`#`, `##`, `###`) and list structure that the chunker's `RecursiveCharacterTextSplitter` can use as natural split boundaries, resulting in semantically coherent chunks.

---

## Section 5: Error Handling & Resume

| Scenario | Behavior |
|---|---|
| HTTP non-200 | Mark `failed`, log warning, continue |
| Network timeout / exception | Tenacity retries (3 attempts), then mark `failed` |
| Empty markdown after cleaning | Mark `skipped` |
| Duplicate content hash | Mark `skipped` (dedup) |
| Process crash | All unprocessed URLs remain `pending`; next run resumes automatically |
| `--force-rescrape --school X` | Deletes all queue/pages rows for school X, re-crawls from scratch |

---

## Out of Scope

- Parallel school crawling (sequential is fine for now)
- Scrapy framework migration
- Changes to ingestion pipeline, chunker, or vector store
