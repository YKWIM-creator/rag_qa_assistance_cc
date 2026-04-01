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
async def test_fetch_sitemap_urls_returns_urls():
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
