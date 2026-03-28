import pytest
from src.ingestion.chunker import chunk_page
from src.scraper.spider import ScrapedPage


def make_page(text: str, url: str = "http://test.com", school: str = "test") -> ScrapedPage:
    return ScrapedPage(url=url, school=school, text=text, title="Test Page")


def test_short_text_produces_one_chunk():
    page = make_page("This is a short text.")
    chunks = chunk_page(page)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    long_text = "Word " * 600  # ~600 words, should exceed 500 token chunk size
    page = make_page(long_text)
    chunks = chunk_page(page)
    assert len(chunks) > 1


def test_chunks_contain_metadata():
    page = make_page("Some content here.", url="http://baruch.cuny.edu/admissions", school="baruch")
    chunks = chunk_page(page)
    assert chunks[0]["metadata"]["url"] == "http://baruch.cuny.edu/admissions"
    assert chunks[0]["metadata"]["school"] == "baruch"
    assert "chunk_index" in chunks[0]["metadata"]


def test_chunks_have_text_field():
    page = make_page("Hello world content.")
    chunks = chunk_page(page)
    assert "text" in chunks[0]
    assert len(chunks[0]["text"]) > 0


def test_empty_text_returns_no_chunks():
    page = make_page("")
    chunks = chunk_page(page)
    assert chunks == []
