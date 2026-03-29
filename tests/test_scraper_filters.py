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
