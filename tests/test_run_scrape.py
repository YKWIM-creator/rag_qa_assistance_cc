import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_force_rescrape_calls_clear_school(tmp_path):
    """When --force-rescrape is passed, ScraperDB.clear_school is called before crawling."""
    with patch("scripts.run_scrape.crawl_school", new_callable=AsyncMock) as mock_crawl, \
         patch("scripts.run_scrape.ingest_pages", new_callable=AsyncMock), \
         patch("scripts.run_scrape.get_embedding_model", return_value=MagicMock()), \
         patch("scripts.run_scrape.ScraperDB") as mock_db_class:

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_crawl.return_value = []

        from scripts.run_scrape import main
        await main(school_filter="baruch", max_pages=5, force_rescrape=True)

        mock_db.clear_school.assert_called_once_with("baruch")
        mock_crawl.assert_called_once()
