def test_scraper_db_path_default():
    from config.settings import settings
    assert settings.scraper_db_path == "./scraper_cache/scraper.db"
