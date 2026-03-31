# RAG Quality Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve RAG answer quality by adding Markdown-aware chunking, page type classification, sitemap seeding, query rewriting with school detection, and section-aware prompt formatting.

**Architecture:** Built on top of `feat/scraper-redesign`. The scraper gains sitemap seeding and page type classification; the chunker becomes Markdown-aware with enriched metadata; a new `rewriter.py` extracts school + rewrites queries before retrieval; `chain.py` gains section-prefixed context and an updated prompt.

**Tech Stack:** Python, LangChain `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`, SQLite (`sqlite3`), httpx, BeautifulSoup, `xml.etree.ElementTree` (stdlib for sitemap parsing)

**Branch:** Create from `feat/scraper-redesign` (not `main`) — run `git checkout feat/scraper-redesign && git checkout -b feat/rag-quality-improvements`

---

### Task 1: Add `page_type` column to ScraperDB

**Files:**
- Modify: `src/scraper/db.py`
- Modify: `tests/test_scraper_db.py`

**Step 1: Write the failing test**

Add to `tests/test_scraper_db.py`:

```python
def test_save_page_with_page_type(db):
    db.save_page(
        url="http://example.com/admissions",
        school="test",
        title="Admissions",
        markdown="# Admissions",
        content_hash="h1",
        page_type="admissions",
    )
    pages = db.get_pages_for_school("test")
    assert pages[0]["page_type"] == "admissions"


def test_save_page_defaults_page_type_to_general(db):
    db.save_page("http://example.com/x", "test", "X", "# X", "h2")
    pages = db.get_pages_for_school("test")
    assert pages[0]["page_type"] == "general"
```

**Step 2: Run tests to verify they fail**

```bash
source ../../.venv/bin/activate && pytest tests/test_scraper_db.py::test_save_page_with_page_type tests/test_scraper_db.py::test_save_page_defaults_page_type_to_general -v
```

Expected: FAIL — `save_page()` doesn't accept `page_type`

**Step 3: Update `src/scraper/db.py`**

Add `page_type TEXT DEFAULT 'general'` to the `pages` table in `_create_tables()`:

```python
def _create_tables(self):
    self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS queue (
            url      TEXT PRIMARY KEY,
            school   TEXT NOT NULL,
            status   TEXT NOT NULL DEFAULT 'pending',
            added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pages (
            url          TEXT PRIMARY KEY,
            school       TEXT NOT NULL,
            title        TEXT,
            markdown     TEXT,
            content_hash TEXT,
            page_type    TEXT NOT NULL DEFAULT 'general',
            scraped_at   TEXT
        );
    """)
    self.conn.commit()
```

Update `save_page()` to accept and store `page_type`:

```python
def save_page(self, url: str, school: str, title: str, markdown: str,
              content_hash: str, page_type: str = "general"):
    now = datetime.now(timezone.utc).isoformat()
    self.conn.execute(
        """INSERT OR REPLACE INTO pages
           (url, school, title, markdown, content_hash, page_type, scraped_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (url, school, title, markdown, content_hash, page_type, now),
    )
    self.conn.commit()
```

**Step 4: Run tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_scraper_db.py -v
```

Expected: all PASS

**Step 5: Run full suite**

```bash
source ../../.venv/bin/activate && pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass

**Step 6: Commit**

```bash
git add src/scraper/db.py tests/test_scraper_db.py
git commit -m "feat: add page_type column to ScraperDB pages table"
```

---

### Task 2: Create `src/scraper/classifier.py` — page type classifier

**Files:**
- Create: `src/scraper/classifier.py`
- Create: `tests/test_scraper_classifier.py`

**Step 1: Write the failing tests**

Create `tests/test_scraper_classifier.py`:

```python
import pytest
from src.scraper.classifier import classify_page


@pytest.mark.parametrize("url,h1,expected", [
    ("https://baruch.cuny.edu/admissions/apply", "Apply Now", "admissions"),
    ("https://baruch.cuny.edu/admissions/requirements", "Requirements", "admissions"),
    ("https://hunter.cuny.edu/academics/programs", "Programs", "academics"),
    ("https://hunter.cuny.edu/departments/math", "Math Department", "academics"),
    ("https://qc.cuny.edu/financial-aid/scholarships", "Scholarships", "financial_aid"),
    ("https://qc.cuny.edu/tuition", "Tuition & Fees", "financial_aid"),
    ("https://brooklyn.cuny.edu/housing", "Student Housing", "student_services"),
    ("https://brooklyn.cuny.edu/student-affairs/advising", "Academic Advising", "student_services"),
    ("https://baruch.cuny.edu/about", "About Baruch", "general"),
    ("https://baruch.cuny.edu/news/2024", "News", "general"),
])
def test_classify_page(url, h1, expected):
    assert classify_page(url, h1) == expected


def test_classify_page_empty_h1():
    result = classify_page("https://baruch.cuny.edu/admissions", "")
    assert result == "admissions"


def test_classify_page_no_signals():
    result = classify_page("https://baruch.cuny.edu/contact", "Contact Us")
    assert result == "general"
```

**Step 2: Run tests to verify they fail**

```bash
source ../../.venv/bin/activate && pytest tests/test_scraper_classifier.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create `src/scraper/classifier.py`**

```python
from urllib.parse import urlparse

_URL_SIGNALS = {
    "admissions": ["/admissions", "/apply", "/requirements", "/enrollment", "/undergraduate", "/graduate"],
    "academics": ["/academics", "/programs", "/majors", "/departments", "/courses", "/curriculum", "/degrees"],
    "financial_aid": ["/financial", "/aid", "/scholarships", "/tuition", "/fees", "/grants", "/loans"],
    "student_services": ["/housing", "/advising", "/health", "/career", "/clubs", "/student-life", "/student-affairs"],
}

_H1_SIGNALS = {
    "admissions": ["admissions", "apply", "requirements", "enrollment"],
    "academics": ["academics", "programs", "majors", "departments", "courses", "curriculum"],
    "financial_aid": ["financial", "aid", "scholarship", "tuition", "fees", "grant"],
    "student_services": ["housing", "advising", "health", "career", "clubs"],
}


def classify_page(url: str, h1_text: str) -> str:
    """Classify a page into one of 5 content types based on URL and h1 text."""
    path = urlparse(url.lower()).path
    h1 = h1_text.lower()

    for page_type, signals in _URL_SIGNALS.items():
        for signal in signals:
            if signal in path:
                return page_type

    for page_type, signals in _H1_SIGNALS.items():
        for signal in signals:
            if signal in h1:
                return page_type

    return "general"
```

**Step 4: Run tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_scraper_classifier.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/scraper/classifier.py tests/test_scraper_classifier.py
git commit -m "feat: add page type classifier for admissions/academics/financial_aid/student_services"
```

---

### Task 3: Add `page_type` to `ScrapedPage` + sitemap seeding in `spider.py`

**Files:**
- Modify: `src/scraper/spider.py`
- Modify: `tests/test_spider.py`

**Step 1: Update `tests/test_spider.py`**

Replace the file with:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.scraper.spider import ScrapedPage, crawl_school


def test_scraped_page_has_required_fields():
    page = ScrapedPage(url="http://x.com", school="baruch", text="text", title="Title")
    assert page.url == "http://x.com"
    assert page.school == "baruch"
    assert page.text == "text"
    assert page.page_type == "general"


def test_scraped_page_accepts_page_type():
    page = ScrapedPage(url="http://x.com", school="baruch", text="# Hi",
                       title="T", page_type="admissions")
    assert page.page_type == "admissions"


@pytest.mark.asyncio
async def test_crawl_school_returns_scraped_pages(tmp_path):
    db_path = str(tmp_path / "test.db")
    html = ("<html><head><title>Admissions</title></head>"
            "<body><main><h1>Admissions</h1><p>Apply here.</p></main></body></html>")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html

    with patch("src.scraper.spider.settings") as mock_settings, \
         patch("src.scraper.spider.httpx.AsyncClient") as mock_client_class, \
         patch("src.scraper.spider.asyncio.sleep", new_callable=AsyncMock), \
         patch("src.scraper.spider.fetch_sitemap_urls", new_callable=AsyncMock, return_value=[]):

        mock_settings.scraper_timeout = 10
        mock_settings.scraper_rate_limit_delay = 0
        mock_settings.scraper_db_path = db_path

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        pages = await crawl_school("baruch", "http://example.com", max_pages=1)

    assert len(pages) == 1
    assert pages[0].school == "baruch"
    assert "Admissions" in pages[0].text
    assert pages[0].page_type in ("admissions", "general")


@pytest.mark.asyncio
async def test_crawl_school_skips_failed_urls(tmp_path):
    db_path = str(tmp_path / "test.db")

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("src.scraper.spider.settings") as mock_settings, \
         patch("src.scraper.spider.httpx.AsyncClient") as mock_client_class, \
         patch("src.scraper.spider.asyncio.sleep", new_callable=AsyncMock), \
         patch("src.scraper.spider.fetch_sitemap_urls", new_callable=AsyncMock, return_value=[]):

        mock_settings.scraper_timeout = 10
        mock_settings.scraper_rate_limit_delay = 0
        mock_settings.scraper_db_path = db_path

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        pages = await crawl_school("baruch", "http://example.com", max_pages=5)

    assert pages == []


@pytest.mark.asyncio
async def test_fetch_sitemap_urls_returns_urls(tmp_path):
    from src.scraper.spider import fetch_sitemap_urls

    sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://baruch.cuny.edu/admissions</loc></url>
        <url><loc>https://baruch.cuny.edu/academics</loc></url>
    </urlset>"""

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sitemap_xml

    mock_404 = MagicMock()
    mock_404.status_code = 404

    with patch("src.scraper.spider.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[mock_response, mock_404])
        mock_client_class.return_value = mock_client

        urls = await fetch_sitemap_urls("https://baruch.cuny.edu", timeout=10)

    assert "https://baruch.cuny.edu/admissions" in urls
    assert "https://baruch.cuny.edu/academics" in urls
```

**Step 2: Run tests to verify they fail**

```bash
source ../../.venv/bin/activate && pytest tests/test_spider.py -v
```

Expected: FAIL — `ScrapedPage` missing `page_type`, `fetch_sitemap_urls` not found

**Step 3: Update `src/scraper/spider.py`**

```python
import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config.settings import settings
from src.scraper.classifier import classify_page
from src.scraper.cleaner import clean_to_markdown
from src.scraper.db import ScraperDB
from src.scraper.filters import should_skip_url

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    url: str
    school: str
    text: str        # set to markdown — backward-compatible with ingestion pipeline
    title: str = ""
    page_type: str = "general"
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


async def fetch_sitemap_urls(base_url: str, timeout: int = 10) -> list[str]:
    """Fetch and parse sitemap.xml; return all <loc> URLs. Returns [] on failure."""
    sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
    robots_url = base_url.rstrip("/") + "/robots.txt"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # Try sitemap.xml directly
        try:
            resp = await client.get(sitemap_url)
            if resp.status_code == 200:
                return _parse_sitemap(resp.text)
        except Exception:
            pass

        # Fall back: check robots.txt for Sitemap: directive
        try:
            resp = await client.get(robots_url)
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        alt_url = line.split(":", 1)[1].strip()
                        r2 = await client.get(alt_url)
                        if r2.status_code == 200:
                            return _parse_sitemap(r2.text)
        except Exception:
            pass

    return []


def _parse_sitemap(xml_text: str) -> list[str]:
    """Extract all <loc> URLs from sitemap XML."""
    try:
        root = ET.fromstring(xml_text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]
    except ET.ParseError:
        return []


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all internal links from a page."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            links.append(full_url.split("#")[0])
    return list(set(links))


async def crawl_school(school: str, start_url: str, max_pages: int = 500) -> list[ScrapedPage]:
    """BFS crawl of a single CUNY school site with SQLite-backed queue."""
    db = ScraperDB(settings.scraper_db_path)

    # Seed from sitemap first, then homepage
    sitemap_urls = await fetch_sitemap_urls(start_url, timeout=settings.scraper_timeout)
    seed_urls = list(dict.fromkeys([start_url] + sitemap_urls))  # deduplicated, homepage first
    db.enqueue_new(seed_urls, school)

    results: list[ScrapedPage] = []

    async with httpx.AsyncClient(timeout=settings.scraper_timeout, follow_redirects=True) as client:
        while len(results) < max_pages:
            url = db.next_pending()
            if url is None:
                break

            if should_skip_url(url):
                db.mark(url, "skipped")
                continue

            await asyncio.sleep(settings.scraper_rate_limit_delay)

            try:
                response = await client.get(url)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                db.mark(url, "failed")
                continue

            if response.status_code != 200:
                logger.warning(f"Skipping {url}: HTTP {response.status_code}")
                db.mark(url, "failed")
                continue

            html = response.text
            markdown = clean_to_markdown(html, url=url)

            if not markdown.strip():
                db.mark(url, "skipped")
                continue

            content_hash = hashlib.sha256(markdown.encode()).hexdigest()
            if db.hash_exists(content_hash):
                logger.debug(f"Duplicate content, skipping: {url}")
                db.mark(url, "skipped")
                continue

            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            h1_tag = soup.find("h1")
            h1_text = h1_tag.get_text(strip=True) if h1_tag else ""
            page_type = classify_page(url, h1_text)

            db.save_page(url, school, title, markdown, content_hash, page_type=page_type)
            db.mark(url, "scraped")

            results.append(ScrapedPage(
                url=url, school=school, text=markdown,
                title=title, page_type=page_type,
            ))
            logger.info(f"[{school}] Scraped ({page_type}): {url}")

            new_links = extract_links(html, url)
            db.enqueue_new(new_links, school)

    logger.info(f"[{school}] Done: {len(results)} pages")
    db.close()
    return results
```

**Step 4: Run spider tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_spider.py -v
```

Expected: all 5 tests PASS

**Step 5: Run full suite**

```bash
source ../../.venv/bin/activate && pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass

**Step 6: Commit**

```bash
git add src/scraper/spider.py tests/test_spider.py
git commit -m "feat: add sitemap seeding and page_type classification to spider"
```

---

### Task 4: Two-pass Markdown-aware chunker + enriched metadata

**Files:**
- Modify: `src/ingestion/chunker.py`
- Modify: `tests/test_chunker.py`

**Step 1: Update `tests/test_chunker.py`**

Replace the file with:

```python
import pytest
from src.ingestion.chunker import chunk_page
from src.scraper.spider import ScrapedPage


def make_page(text: str, url: str = "http://test.com", school: str = "test",
              title: str = "Test Page", page_type: str = "general") -> ScrapedPage:
    return ScrapedPage(url=url, school=school, text=text, title=title, page_type=page_type)


def test_short_text_produces_one_chunk():
    page = make_page("This is a short text.")
    chunks = chunk_page(page)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    long_text = "Word " * 600
    page = make_page(long_text)
    chunks = chunk_page(page)
    assert len(chunks) > 1


def test_chunks_contain_base_metadata():
    page = make_page("Some content.", url="http://baruch.cuny.edu/admissions",
                     school="baruch", page_type="admissions")
    chunks = chunk_page(page)
    meta = chunks[0]["metadata"]
    assert meta["url"] == "http://baruch.cuny.edu/admissions"
    assert meta["school"] == "baruch"
    assert meta["page_type"] == "admissions"
    assert "chunk_index" in meta


def test_chunks_contain_section_heading():
    md = "# Welcome\n\n## Admissions Requirements\n\nYou must submit transcripts.\n"
    page = make_page(md, title="Admissions | Baruch")
    chunks = chunk_page(page)
    headings = [c["metadata"]["section_heading"] for c in chunks]
    assert any("Admissions Requirements" in h for h in headings)


def test_section_heading_falls_back_to_title():
    page = make_page("No headings here, just plain text.", title="About Baruch")
    chunks = chunk_page(page)
    assert chunks[0]["metadata"]["section_heading"] == "About Baruch"


def test_chunks_have_text_field():
    page = make_page("Hello world content.")
    chunks = chunk_page(page)
    assert "text" in chunks[0]
    assert len(chunks[0]["text"]) > 0


def test_empty_text_returns_no_chunks():
    page = make_page("")
    chunks = chunk_page(page)
    assert chunks == []


def test_markdown_header_split_keeps_section_together():
    md = "## Requirements\n\n" + "requirement detail " * 20 + "\n\n## Deadlines\n\n" + "deadline info " * 20
    page = make_page(md)
    chunks = chunk_page(page)
    # No chunk should contain content from both sections
    for chunk in chunks:
        text = chunk["text"]
        has_req = "requirement detail" in text
        has_dead = "deadline info" in text
        assert not (has_req and has_dead), f"Chunk spans two sections: {text[:80]}"
```

**Step 2: Run tests to verify they fail**

```bash
source ../../.venv/bin/activate && pytest tests/test_chunker.py -v
```

Expected: several FAIL — `section_heading` and `page_type` not in metadata

**Step 3: Rewrite `src/ingestion/chunker.py`**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.scraper.spider import ScrapedPage

_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_page(page: ScrapedPage) -> list[dict]:
    """Split a ScrapedPage into chunks with enriched metadata."""
    if not page.text.strip():
        return []

    # Pass 1: split on Markdown headers
    header_docs = _HEADER_SPLITTER.split_text(page.text)

    # Pass 2: character-split any section that's still too large
    final_chunks = []
    for doc in header_docs:
        sub_texts = _CHAR_SPLITTER.split_text(doc.page_content)
        for sub in sub_texts:
            final_chunks.append((sub, doc.metadata))

    chunks = []
    for i, (text, header_meta) in enumerate(final_chunks):
        # Build section_heading: prefer deepest header found, fallback to page title
        section_heading = (
            header_meta.get("h3")
            or header_meta.get("h2")
            or header_meta.get("h1")
            or page.title
        )
        chunks.append({
            "text": text,
            "metadata": {
                "url": page.url,
                "school": page.school,
                "title": page.title,
                "section_heading": section_heading,
                "page_type": page.page_type,
                "chunk_index": i,
                "scraped_at": page.scraped_at,
            },
        })
    return chunks
```

**Step 4: Run chunker tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_chunker.py -v
```

Expected: all PASS

**Step 5: Run full suite**

```bash
source ../../.venv/bin/activate && pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass

**Step 6: Commit**

```bash
git add src/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: two-pass Markdown-aware chunker with section_heading and page_type metadata"
```

---

### Task 5: Retriever — raise `fetch_k` and add metadata filter support

**Files:**
- Modify: `src/retrieval/retriever.py`
- Modify: `tests/test_retriever.py`

**Step 1: Read `tests/test_retriever.py` to understand existing tests**

```bash
cat tests/test_retriever.py
```

**Step 2: Add a failing test**

Append to `tests/test_retriever.py`:

```python
def test_get_retriever_accepts_metadata_filter():
    from unittest.mock import MagicMock
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    retriever = get_retriever(mock_vs, k=5, metadata_filter={"school": "baruch"})
    call_kwargs = mock_vs.as_retriever.call_args[1]
    assert call_kwargs["search_kwargs"]["filter"] == {"school": "baruch"}
    assert call_kwargs["search_kwargs"]["fetch_k"] == 25
```

**Step 3: Run test to verify it fails**

```bash
source ../../.venv/bin/activate && pytest tests/test_retriever.py::test_get_retriever_accepts_metadata_filter -v
```

Expected: FAIL

**Step 4: Update `src/retrieval/retriever.py`**

Update `get_retriever()`:

```python
def get_retriever(vectorstore: Chroma, k: int = 5, metadata_filter: dict = None):
    """Return a LangChain MMR retriever from a ChromaDB vector store."""
    search_kwargs = {"k": k, "fetch_k": k * 5}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )
```

**Step 5: Run retriever tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_retriever.py -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add src/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: raise fetch_k to k*5 and add optional metadata_filter to get_retriever"
```

---

### Task 6: Create `src/generation/rewriter.py` — query rewriting with school detection

**Files:**
- Create: `src/generation/rewriter.py`
- Create: `tests/test_rewriter.py`

**Step 1: Write the failing tests**

Create `tests/test_rewriter.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from src.generation.rewriter import rewrite_query, RewrittenQuery


def test_rewritten_query_dataclass():
    q = RewrittenQuery(query="admissions requirements", school="baruch")
    assert q.query == "admissions requirements"
    assert q.school == "baruch"


def test_rewritten_query_no_school():
    q = RewrittenQuery(query="what are CUNY programs?", school=None)
    assert q.school is None


def test_rewrite_query_calls_llm(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"school": "baruch", "query": "Baruch admissions requirements"}'

    result = rewrite_query("how do i get into baruch", mock_llm)

    assert isinstance(result, RewrittenQuery)
    assert result.school == "baruch"
    assert "baruch" in result.query.lower()
    mock_llm.invoke.assert_called_once()


def test_rewrite_query_handles_no_school(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"school": null, "query": "CUNY financial aid options"}'

    result = rewrite_query("what financial aid does cuny offer", mock_llm)

    assert result.school is None
    assert result.query == "CUNY financial aid options"


def test_rewrite_query_falls_back_on_bad_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "I cannot parse this"

    result = rewrite_query("original question", mock_llm)

    assert result.query == "original question"
    assert result.school is None
```

**Step 2: Run tests to verify they fail**

```bash
source ../../.venv/bin/activate && pytest tests/test_rewriter.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create `src/generation/rewriter.py`**

```python
import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You are a query preprocessing assistant for a CUNY university information system.

Given a user question, return a JSON object with:
- "school": the CUNY school short name if the question targets a specific school, or null if general.
  Valid school names: baruch, brooklyn, city, hunter, john_jay, lehman, medgar_evers, nycct, queens, staten_island, york
- "query": a rewritten, retrieval-optimized version of the question (more specific, academic language)

Question: {question}

Return ONLY valid JSON. Example: {{"school": "baruch", "query": "Baruch College undergraduate admissions requirements GPA"}}"""


@dataclass
class RewrittenQuery:
    query: str
    school: Optional[str]


def rewrite_query(question: str, llm) -> RewrittenQuery:
    """Extract school + rewrite question for better retrieval. Falls back gracefully."""
    prompt = _REWRITE_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code fences if present
        content = content.strip().strip("```json").strip("```").strip()
        data = json.loads(content)
        return RewrittenQuery(
            query=data.get("query", question),
            school=data.get("school") or None,
        )
    except Exception as e:
        logger.warning(f"Query rewriting failed, using original: {e}")
        return RewrittenQuery(query=question, school=None)
```

**Step 4: Run tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_rewriter.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add src/generation/rewriter.py tests/test_rewriter.py
git commit -m "feat: add query rewriter with school detection and graceful fallback"
```

---

### Task 7: Update `chain.py` — richer context format + query rewriting + new prompt

**Files:**
- Modify: `src/generation/chain.py`
- Modify: `tests/test_chain.py`

**Step 1: Add failing tests to `tests/test_chain.py`**

Append:

```python
def test_format_docs_includes_provenance():
    from src.generation.chain import _format_docs
    from unittest.mock import MagicMock
    doc = MagicMock()
    doc.page_content = "Apply by December 1st."
    doc.metadata = {
        "school": "baruch",
        "page_type": "admissions",
        "section_heading": "Application Deadlines",
    }
    result = _format_docs([doc])
    assert "Baruch" in result or "baruch" in result
    assert "admissions" in result
    assert "Application Deadlines" in result
    assert "Apply by December 1st." in result


def test_ask_uses_rewriter(monkeypatch):
    from src.generation.chain import ask
    from unittest.mock import MagicMock, patch

    mock_doc = MagicMock()
    mock_doc.page_content = "Baruch admissions info."
    mock_doc.metadata = {"url": "http://baruch.cuny.edu", "school": "baruch",
                         "title": "Admissions", "page_type": "admissions",
                         "section_heading": "Requirements"}

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"school": "baruch", "query": "Baruch admissions"}'

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "You need a 3.0 GPA."

    with patch("src.generation.chain.build_rag_chain", return_value=mock_chain), \
         patch("src.generation.chain.rewrite_query") as mock_rw:
        from src.generation.rewriter import RewrittenQuery
        mock_rw.return_value = RewrittenQuery(query="Baruch admissions", school="baruch")
        result = ask("how do i get into baruch", mock_retriever, mock_llm)

    assert result.answer == "You need a 3.0 GPA."
    mock_rw.assert_called_once()
```

**Step 2: Run tests to verify they fail**

```bash
source ../../.venv/bin/activate && pytest tests/test_chain.py::test_format_docs_includes_provenance tests/test_chain.py::test_ask_uses_rewriter -v
```

Expected: FAIL

**Step 3: Update `src/generation/chain.py`**

```python
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from src.generation.rewriter import rewrite_query


PROMPT_TEMPLATE = """You are a CUNY student assistant. Answer the question using ONLY the context below.
Each context block is labeled with [School | page type | section].
If the question asks about a specific school, prioritize blocks from that school.
If the answer is not found in the context, say: "I don't have information about that in the CUNY documents I've indexed."
Be concise and cite the school name in your answer.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        school = doc.metadata.get("school", "unknown").title()
        page_type = doc.metadata.get("page_type", "general")
        heading = doc.metadata.get("section_heading", doc.metadata.get("title", ""))
        label = f"[{school} | {page_type} | {heading}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(retriever, llm):
    """Build a LangChain RAG chain from a retriever and LLM."""
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, retriever, llm) -> RAGResponse:
    """Run RAG pipeline: rewrite query → retrieve → generate → return answer + sources."""
    # Rewrite query and extract school
    rewritten = rewrite_query(question, llm)

    # Apply school filter if detected
    if rewritten.school and hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["filter"] = {"school": rewritten.school}

    docs = retriever.invoke(rewritten.query)

    if not docs:
        return RAGResponse(
            answer="I don't have information about that in the CUNY documents I've indexed.",
            sources=[],
        )

    chain = build_rag_chain(retriever, llm)
    answer = chain.invoke(rewritten.query)

    sources = [
        {
            "url": doc.metadata.get("url", ""),
            "school": doc.metadata.get("school", ""),
            "title": doc.metadata.get("title", ""),
        }
        for doc in docs
    ]
    seen_urls: set[str] = set()
    unique_sources = []
    for s in sources:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            unique_sources.append(s)

    return RAGResponse(answer=answer, sources=unique_sources)
```

**Step 4: Run chain tests**

```bash
source ../../.venv/bin/activate && pytest tests/test_chain.py -v
```

Expected: all PASS

**Step 5: Run full suite**

```bash
source ../../.venv/bin/activate && pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass

**Step 6: Commit**

```bash
git add src/generation/chain.py tests/test_chain.py
git commit -m "feat: section-aware context format, query rewriting, and updated RAG prompt"
```

---

### Task 8: Final verification

**Step 1: Run complete test suite**

```bash
source ../../.venv/bin/activate && pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests PASS (60+ tests)

**Step 2: Verify all new modules import cleanly**

```bash
source ../../.venv/bin/activate && python -c "
from src.scraper.classifier import classify_page
from src.scraper.spider import ScrapedPage, fetch_sitemap_urls
from src.ingestion.chunker import chunk_page
from src.retrieval.retriever import get_retriever
from src.generation.rewriter import rewrite_query, RewrittenQuery
from src.generation.chain import ask, _format_docs
print('All imports OK')
"
```

Expected: `All imports OK`

**Step 3: Smoke test — chunk a Markdown page and verify enriched metadata**

```bash
source ../../.venv/bin/activate && python -c "
from src.scraper.spider import ScrapedPage
from src.ingestion.chunker import chunk_page

md = '''# Baruch College Admissions

## Undergraduate Requirements

Applicants must have a minimum GPA of 3.0 and submit SAT scores.

## Transfer Requirements

Transfer students need 24+ credits with a 2.8 GPA.
'''

page = ScrapedPage(url='http://baruch.cuny.edu/admissions', school='baruch',
                   text=md, title='Admissions | Baruch', page_type='admissions')
chunks = chunk_page(page)
for c in chunks:
    print(c['metadata']['section_heading'], '|', c['text'][:60])
"
```

Expected: each chunk shows its `section_heading` (e.g. `Undergraduate Requirements | Applicants must...`)

**Step 4: Check git log**

```bash
git log --oneline -8
```

Expected: 7 feature commits visible on top of the `feat/scraper-redesign` history
