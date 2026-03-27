import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.scraper.spider import scrape_page, ScrapedPage


@pytest.mark.asyncio
async def test_scrape_page_returns_scraped_page_on_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><head><title>Test</title></head><body><main><p>Hello</p></main></body></html>"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await scrape_page("http://example.com", school="test")

    assert isinstance(result, ScrapedPage)
    assert result.url == "http://example.com"
    assert result.school == "test"
    assert "Hello" in result.text


@pytest.mark.asyncio
async def test_scrape_page_returns_none_on_404():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await scrape_page("http://example.com/notfound", school="test")

    assert result is None


def test_scraped_page_has_required_fields():
    page = ScrapedPage(url="http://x.com", school="baruch", text="text", title="Title")
    assert page.url == "http://x.com"
    assert page.school == "baruch"
