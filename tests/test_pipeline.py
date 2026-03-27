import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.ingestion.pipeline import ingest_pages


@pytest.mark.asyncio
async def test_ingest_pages_stores_chunks():
    from src.scraper.spider import ScrapedPage

    pages = [
        ScrapedPage(url="http://baruch.cuny.edu", school="baruch", text="Baruch offers business programs.", title="Home"),
    ]

    mock_embeddings = MagicMock()
    mock_vectorstore = MagicMock()

    with patch("src.ingestion.pipeline.build_vectorstore", return_value=mock_vectorstore) as mock_build:
        result = await ingest_pages(pages, mock_embeddings)

    mock_build.assert_called_once()
    assert result == mock_vectorstore
