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
         patch("src.scraper.spider.httpx.AsyncClient") as mock_client_class, \
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
         patch("src.scraper.spider.httpx.AsyncClient") as mock_client_class, \
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
