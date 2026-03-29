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
