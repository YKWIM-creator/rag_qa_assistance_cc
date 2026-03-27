import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.scraper.cleaner import clean_html

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    url: str
    school: str
    text: str
    title: str = ""
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@retry(
    stop=stop_after_attempt(settings.scraper_max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=False,
)
async def scrape_page(url: str, school: str) -> Optional[ScrapedPage]:
    """Fetch a single URL and return a ScrapedPage, or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=settings.scraper_timeout, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.warning(f"Skipping {url}: HTTP {response.status_code}")
                return None

            text = clean_html(response.text, url=url)
            if not text.strip():
                return None

            soup = BeautifulSoup(response.text, "lxml")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            return ScrapedPage(url=url, school=school, text=text, title=title)

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all internal links from a page."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only same-domain, http/https, no fragments
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            links.append(full_url.split("#")[0])  # strip fragments

    return list(set(links))


async def crawl_school(school: str, start_url: str, max_pages: int = 500) -> list[ScrapedPage]:
    """BFS crawl of a single CUNY school site."""
    seen: set[str] = set()
    queue: list[str] = [start_url]
    results: list[ScrapedPage] = []

    async with httpx.AsyncClient(timeout=settings.scraper_timeout, follow_redirects=True) as client:
        while queue and len(results) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            await asyncio.sleep(settings.scraper_rate_limit_delay)

            try:
                response = await client.get(url)
                if response.status_code != 200:
                    continue

                page = await scrape_page(url, school)
                if page:
                    results.append(page)
                    logger.info(f"[{school}] Scraped: {url}")

                new_links = extract_links(response.text, url)
                for link in new_links:
                    if link not in seen:
                        queue.append(link)

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                continue

    logger.info(f"[{school}] Done: {len(results)} pages")
    return results
