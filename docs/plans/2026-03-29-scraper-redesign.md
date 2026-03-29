# Scraper Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the CUNY web scraper to persist crawl state in SQLite, output Markdown via markdownify, filter low-value URLs, and deduplicate content — while keeping `ScrapedPage.text` backward-compatible with the ingestion pipeline.

**Architecture:** A new `ScraperDB` class wraps a SQLite file with `pages` and `queue` tables, making every crawl resumable after a crash. The BFS loop in `crawl_school` is rewritten to read/write this DB and make a single HTTP fetch per URL. `cleaner.py` replaces `get_text()` with `markdownify()` to preserve heading and list structure.

**Tech Stack:** Python stdlib `sqlite3`, `markdownify` (new dep), `httpx`, `beautifulsoup4`, `tenacity`

---

### Task 1: Add `markdownify` dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add the dependency**

Append to `requirements.txt`:
```
markdownify==0.12.1
```

**Step 2: Install it**

```bash
uv pip install markdownify==0.12.1
```

Expected output: `Successfully installed markdownify-0.12.1`

**Step 3: Verify import works**

```bash
python -c "import markdownify; print('ok')"
```

Expected: `ok`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add markdownify dependency"
```

---

### Task 2: Add `scraper_db_path` to settings

**Files:**
- Modify: `config/settings.py`

**Step 1: Write the failing test**

In `tests/test_settings.py` (create if it doesn't exist — check first with `ls tests/`):

```python
def test_scraper_db_path_default():
    from config.settings import settings
    assert settings.scraper_db_path == "./scraper_cache/scraper.db"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_settings.py::test_scraper_db_path_default -v
```

Expected: FAIL with `AttributeError`

**Step 3: Add the field to `config/settings.py`**

In the `# Scraper` section, after `scraper_timeout`, add:
```python
scraper_db_path: str = Field(default="./scraper_cache/scraper.db", env="SCRAPER_DB_PATH")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_settings.py::test_scraper_db_path_default -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add config/settings.py tests/test_settings.py
git commit -m "feat: add scraper_db_path setting"
```

---

### Task 3: Create `src/scraper/db.py` — ScraperDB

**Files:**
- Create: `src/scraper/db.py`
- Create: `tests/test_scraper_db.py`

**Step 1: Write the failing tests**

Create `tests/test_scraper_db.py`:

```python
import pytest
import os
import tempfile
from src.scraper.db import ScraperDB


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = ScraperDB(db_path)
    yield db
    db.close()


def test_enqueue_and_next_pending(db):
    db.enqueue_new(["http://example.com/a", "http://example.com/b"], school="test")
    url = db.next_pending()
    assert url in ("http://example.com/a", "http://example.com/b")


def test_enqueue_is_idempotent(db):
    db.enqueue_new(["http://example.com/a"], school="test")
    db.enqueue_new(["http://example.com/a"], school="test")  # second enqueue ignored
    db.mark(url="http://example.com/a", status="scraped")
    assert db.next_pending() is None


def test_mark_status(db):
    db.enqueue_new(["http://example.com/x"], school="test")
    db.mark("http://example.com/x", "failed")
    assert db.next_pending() is None


def test_save_page_and_hash_exists(db):
    db.save_page(
        url="http://example.com/page",
        school="test",
        title="Test Page",
        markdown="# Hello",
        content_hash="abc123",
    )
    assert db.hash_exists("abc123") is True
    assert db.hash_exists("notexist") is False


def test_get_pages_for_school(db):
    db.save_page("http://x.com/1", "school_a", "T1", "# A", "h1")
    db.save_page("http://x.com/2", "school_b", "T2", "# B", "h2")
    pages = db.get_pages_for_school("school_a")
    assert len(pages) == 1
    assert pages[0]["url"] == "http://x.com/1"


def test_clear_school(db):
    db.enqueue_new(["http://x.com/1"], "school_a")
    db.save_page("http://x.com/1", "school_a", "T", "# Hi", "h1")
    db.clear_school("school_a")
    assert db.next_pending() is None
    assert db.get_pages_for_school("school_a") == []
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scraper_db.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.scraper.db'`

**Step 3: Implement `src/scraper/db.py`**

```python
import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional


class ScraperDB:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

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
                scraped_at   TEXT
            );
        """)
        self.conn.commit()

    def enqueue_new(self, urls: list[str], school: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.executemany(
            "INSERT OR IGNORE INTO queue (url, school, status, added_at) VALUES (?, ?, 'pending', ?)",
            [(url, school, now) for url in urls],
        )
        self.conn.commit()

    def next_pending(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT url FROM queue WHERE status = 'pending' LIMIT 1"
        ).fetchone()
        return row["url"] if row else None

    def mark(self, url: str, status: str):
        self.conn.execute("UPDATE queue SET status = ? WHERE url = ?", (status, url))
        self.conn.commit()

    def hash_exists(self, content_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM pages WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row is not None

    def save_page(self, url: str, school: str, title: str, markdown: str, content_hash: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO pages (url, school, title, markdown, content_hash, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (url, school, title, markdown, content_hash, now),
        )
        self.conn.commit()

    def get_pages_for_school(self, school: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM pages WHERE school = ?", (school,)
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_school(self, school: str):
        self.conn.execute("DELETE FROM queue WHERE school = ?", (school,))
        self.conn.execute("DELETE FROM pages WHERE school = ?", (school,))
        self.conn.commit()

    def close(self):
        self.conn.close()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scraper_db.py -v
```

Expected: all 6 tests PASS

**Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS

**Step 6: Commit**

```bash
git add src/scraper/db.py tests/test_scraper_db.py
git commit -m "feat: add ScraperDB with SQLite-backed crawl queue and page store"
```

---

### Task 4: Create `src/scraper/filters.py` — URL filter

**Files:**
- Create: `src/scraper/filters.py`
- Create: `tests/test_scraper_filters.py`

**Step 1: Write the failing tests**

Create `tests/test_scraper_filters.py`:

```python
import pytest
from src.scraper.filters import should_skip_url


@pytest.mark.parametrize("url", [
    "https://example.com/file.pdf",
    "https://example.com/doc.docx",
    "https://example.com/image.png",
    "https://example.com/archive.zip",
    "https://example.com/video.mp4",
])
def test_skips_binary_extensions(url):
    assert should_skip_url(url) is True


@pytest.mark.parametrize("url", [
    "https://example.com/login",
    "https://example.com/logout",
    "https://example.com/calendar/2024",
    "https://example.com/events/spring",
    "https://example.com/page?print=1",
    "https://example.com/wp-json/v2/posts",
    "https://example.com/feed/rss",
])
def test_skips_low_value_paths(url):
    assert should_skip_url(url) is True


@pytest.mark.parametrize("url", [
    "https://www.baruch.cuny.edu/admissions",
    "https://www.hunter.cuny.edu/academics/programs",
    "https://www.brooklyn.cuny.edu/about",
])
def test_allows_normal_pages(url):
    assert should_skip_url(url) is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scraper_filters.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `src/scraper/filters.py`**

```python
from urllib.parse import urlparse

SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".avi", ".mov",
}

SKIP_PATH_FRAGMENTS = [
    "/login", "/logout", "/signin", "/signout",
    "/calendar", "/events",
    "/wp-json", "/feed",
]

SKIP_QUERY_PARAMS = ["print", "sort", "filter", "page"]


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url.lower())
    path = parsed.path

    # Check file extension
    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True

    # Check path fragments
    for fragment in SKIP_PATH_FRAGMENTS:
        if fragment in path:
            return True

    # Check query params
    query = parsed.query
    for param in SKIP_QUERY_PARAMS:
        if f"{param}=" in query:
            return True

    return False
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scraper_filters.py -v
```

Expected: all tests PASS

**Step 5: Commit**

```bash
git add src/scraper/filters.py tests/test_scraper_filters.py
git commit -m "feat: add URL filter to skip binary files and low-value paths"
```

---

### Task 5: Rewrite `src/scraper/cleaner.py` with markdownify

**Files:**
- Modify: `src/scraper/cleaner.py`
- Modify: `tests/test_cleaner.py` (check if it exists with `ls tests/`)

**Step 1: Check existing cleaner tests**

```bash
ls tests/test_cleaner* 2>/dev/null || echo "no cleaner tests"
```

**Step 2: Write the failing tests**

Create or add to `tests/test_cleaner.py`:

```python
from src.scraper.cleaner import clean_to_markdown

def test_extracts_heading_as_markdown():
    html = "<html><body><main><h1>Welcome</h1><p>Hello world</p></main></body></html>"
    result = clean_to_markdown(html, url="http://example.com")
    assert "# Welcome" in result
    assert "Hello world" in result

def test_strips_nav_and_footer():
    html = """<html><body>
        <nav>Skip nav</nav>
        <main><p>Main content</p></main>
        <footer>Footer text</footer>
    </body></html>"""
    result = clean_to_markdown(html, url="http://example.com")
    assert "Skip nav" not in result
    assert "Footer text" not in result
    assert "Main content" in result

def test_returns_empty_for_no_content():
    html = "<html><body></body></html>"
    result = clean_to_markdown(html, url="http://example.com")
    assert result == ""

def test_preserves_list_structure():
    html = "<html><body><main><ul><li>Item 1</li><li>Item 2</li></ul></main></body></html>"
    result = clean_to_markdown(html, url="http://example.com")
    assert "Item 1" in result
    assert "Item 2" in result
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_cleaner.py -v
```

Expected: FAIL (function `clean_to_markdown` not found or wrong output)

**Step 4: Rewrite `src/scraper/cleaner.py`**

```python
from bs4 import BeautifulSoup
from markdownify import markdownify as md

REMOVE_TAGS = ["nav", "footer", "header", "script", "style", "aside", "iframe", "form"]


def clean_to_markdown(html: str, url: str) -> str:
    """Extract main content from HTML and convert to Markdown."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.find("body")
    )

    if not main:
        return ""

    markdown = md(str(main), heading_style="ATX", strip=["a"])
    # Collapse excessive blank lines
    lines = markdown.splitlines()
    cleaned = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()
```

> Note: The old `clean_html` function is replaced by `clean_to_markdown`. The old name is used nowhere else in the codebase (spider.py imports it directly — we'll update that in Task 6).

**Step 5: Run cleaner tests**

```bash
pytest tests/test_cleaner.py -v
```

Expected: all PASS

**Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: The spider test that imports `clean_html` from `cleaner` will now FAIL — that is expected and will be fixed in Task 6.

**Step 7: Commit**

```bash
git add src/scraper/cleaner.py tests/test_cleaner.py
git commit -m "feat: replace get_text() with markdownify in cleaner"
```

---

### Task 6: Rewrite `src/scraper/spider.py` — single-fetch + SQLite queue

**Files:**
- Modify: `src/scraper/spider.py`
- Modify: `tests/test_spider.py`

**Step 1: Read the current `tests/test_spider.py`** to understand what to preserve

The existing tests cover `scrape_page` and `ScrapedPage`. After this task, `scrape_page` is removed (its logic merges into `crawl_school`). The dataclass stays.

**Step 2: Update `tests/test_spider.py`**

Replace the file contents with:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.scraper.spider import ScrapedPage, crawl_school


def test_scraped_page_has_required_fields():
    page = ScrapedPage(url="http://x.com", school="baruch", text="text", title="Title")
    assert page.url == "http://x.com"
    assert page.school == "baruch"
    assert page.text == "text"


def test_scraped_page_text_equals_markdown():
    """text field is set to markdown value for backward compat with ingestion pipeline."""
    page = ScrapedPage(url="http://x.com", school="baruch", text="# Hello", title="T")
    assert page.text == "# Hello"


@pytest.mark.asyncio
async def test_crawl_school_returns_scraped_pages(tmp_path):
    db_path = str(tmp_path / "test.db")
    html = "<html><head><title>Admissions</title></head><body><main><h1>Admissions</h1><p>Apply here.</p></main></body></html>"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html

    with patch("src.scraper.spider.settings") as mock_settings, \
         patch("httpx.AsyncClient") as mock_client_class, \
         patch("src.scraper.spider.asyncio.sleep", new_callable=AsyncMock):

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


@pytest.mark.asyncio
async def test_crawl_school_skips_failed_urls(tmp_path):
    db_path = str(tmp_path / "test.db")

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("src.scraper.spider.settings") as mock_settings, \
         patch("httpx.AsyncClient") as mock_client_class, \
         patch("src.scraper.spider.asyncio.sleep", new_callable=AsyncMock):

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
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_spider.py -v
```

Expected: FAIL (old `scrape_page` import missing, new tests can't run yet)

**Step 4: Rewrite `src/scraper/spider.py`**

```python
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config.settings import settings
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
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    db.enqueue_new([start_url], school)

    results: list[ScrapedPage] = []

    async with httpx.AsyncClient(timeout=settings.scraper_timeout, follow_redirects=True) as client:
        while len(results) < max_pages:
            url = db.next_pending()
            if url is None:
                break

            # Mark in-progress immediately to avoid re-processing on crash
            db.mark(url, "scraped")

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

            db.save_page(url, school, title, markdown, content_hash)
            db.mark(url, "scraped")

            results.append(ScrapedPage(url=url, school=school, text=markdown, title=title))
            logger.info(f"[{school}] Scraped: {url}")

            new_links = extract_links(html, url)
            db.enqueue_new(new_links, school)

    logger.info(f"[{school}] Done: {len(results)} pages")
    db.close()
    return results
```

**Step 5: Run spider tests**

```bash
pytest tests/test_spider.py -v
```

Expected: all PASS

**Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS (including chunker, ingestion, and API tests — `ScrapedPage.text` is unchanged)

**Step 7: Commit**

```bash
git add src/scraper/spider.py tests/test_spider.py
git commit -m "feat: rewrite crawl_school with single-fetch, SQLite queue, and markdownify output"
```

---

### Task 7: Add `--force-rescrape` flag to `scripts/run_scrape.py`

**Files:**
- Modify: `scripts/run_scrape.py`

**Step 1: Write the failing test**

Create `tests/test_run_scrape.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_force_rescrape_calls_clear_school(tmp_path):
    """When --force-rescrape is passed, ScraperDB.clear_school is called before crawling."""
    from src.scraper.db import ScraperDB

    with patch("scripts.run_scrape.crawl_school", new_callable=AsyncMock) as mock_crawl, \
         patch("scripts.run_scrape.ingest_pages", new_callable=AsyncMock), \
         patch("scripts.run_scrape.get_embedding_model", return_value=MagicMock()), \
         patch("src.scraper.spider.ScraperDB") as mock_db_class:

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_crawl.return_value = []

        from scripts.run_scrape import main
        await main(school_filter="baruch", max_pages=5, force_rescrape=True)

        mock_crawl.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_scrape.py -v
```

Expected: FAIL (`main()` missing `force_rescrape` parameter)

**Step 3: Update `scripts/run_scrape.py`**

```python
"""
Run the full CUNY scrape + ingestion pipeline.

Usage:
    python scripts/run_scrape.py                              # scrape all schools
    python scripts/run_scrape.py --school baruch              # scrape one school
    python scripts/run_scrape.py --max-pages 100              # limit pages per school
    python scripts/run_scrape.py --force-rescrape             # clear DB and re-crawl all
    python scripts/run_scrape.py --school baruch --force-rescrape  # re-crawl one school
"""
import asyncio
import argparse
import logging
import sys

sys.path.insert(0, ".")

from config.settings import settings
from src.scraper.spider import crawl_school
from src.scraper.db import ScraperDB
from src.ingestion.pipeline import ingest_pages
from src.ingestion.embedder import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(school_filter: str | None, max_pages: int, force_rescrape: bool = False):
    embedding_model = get_embedding_model()
    schools = settings.cuny_senior_colleges

    if school_filter:
        if school_filter not in schools:
            logger.error(f"Unknown school: {school_filter}. Options: {list(schools.keys())}")
            return
        schools = {school_filter: schools[school_filter]}

    if force_rescrape:
        db = ScraperDB(settings.scraper_db_path)
        for school in schools:
            logger.info(f"Clearing DB for {school}")
            db.clear_school(school)
        db.close()

    all_pages = []
    for school, url in schools.items():
        logger.info(f"Crawling {school}: {url}")
        pages = await crawl_school(school, url, max_pages=max_pages)
        all_pages.extend(pages)
        logger.info(f"Collected {len(pages)} pages from {school}")

    logger.info(f"Total pages: {len(all_pages)}. Starting ingestion...")
    await ingest_pages(all_pages, embedding_model)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--force-rescrape", action="store_true", default=False)
    args = parser.parse_args()
    asyncio.run(main(args.school, args.max_pages, args.force_rescrape))
```

**Step 4: Run test**

```bash
pytest tests/test_run_scrape.py -v
```

Expected: PASS

**Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS

**Step 6: Commit**

```bash
git add scripts/run_scrape.py tests/test_run_scrape.py
git commit -m "feat: add --force-rescrape flag to run_scrape.py"
```

---

### Task 8: Final verification

**Step 1: Run the complete test suite**

```bash
pytest tests/ -v
```

Expected: all 28+ tests PASS (number may be higher after new tests added in this plan)

**Step 2: Smoke test the scraper with one school, small limit**

```bash
python scripts/run_scrape.py --school baruch --max-pages 3
```

Expected: logs showing 3 pages scraped, no errors, `./scraper_cache/scraper.db` created

**Step 3: Verify DB contents**

```bash
python -c "
import sqlite3
conn = sqlite3.connect('./scraper_cache/scraper.db')
print('Pages:', conn.execute('SELECT COUNT(*) FROM pages').fetchone()[0])
print('Queue:', conn.execute('SELECT status, COUNT(*) FROM queue GROUP BY status').fetchall())
conn.close()
"
```

**Step 4: Verify resume works — re-run should skip already-scraped pages**

```bash
python scripts/run_scrape.py --school baruch --max-pages 3
```

Expected: 0 new pages scraped (all already in DB), logs showing skips

**Step 5: Final commit (if any loose ends)**

```bash
git add -A
git status  # verify nothing unexpected
```
