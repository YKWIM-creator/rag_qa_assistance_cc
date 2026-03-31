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
        try:
            resp = await client.get(sitemap_url)
            if resp.status_code == 200:
                return _parse_sitemap(resp.text)
        except Exception:
            pass

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

    sitemap_urls = await fetch_sitemap_urls(start_url, timeout=settings.scraper_timeout)
    seed_urls = list(dict.fromkeys([start_url] + sitemap_urls))
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
