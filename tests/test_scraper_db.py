import pytest
import os
import tempfile
from src.scraper.db import ScraperDB


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = ScraperDB(db_path)
    yield db
    db.close()


def test_enqueue_and_next_pending(db):
    db.enqueue_new(["http://example.com/a", "http://example.com/b"], school="test")
    url = db.next_pending()
    assert url in ("http://example.com/a", "http://example.com/b")


def test_enqueue_is_idempotent(db):
    db.enqueue_new(["http://example.com/a"], school="test")
    db.enqueue_new(["http://example.com/a"], school="test")  # second enqueue ignored
    db.mark(url="http://example.com/a", status="scraped")
    assert db.next_pending() is None


def test_mark_status(db):
    db.enqueue_new(["http://example.com/x"], school="test")
    db.mark("http://example.com/x", "failed")
    assert db.next_pending() is None


def test_save_page_and_hash_exists(db):
    db.save_page(
        url="http://example.com/page",
        school="test",
        title="Test Page",
        markdown="# Hello",
        content_hash="abc123",
    )
    assert db.hash_exists("abc123") is True
    assert db.hash_exists("notexist") is False


def test_get_pages_for_school(db):
    db.save_page("http://x.com/1", "school_a", "T1", "# A", "h1")
    db.save_page("http://x.com/2", "school_b", "T2", "# B", "h2")
    pages = db.get_pages_for_school("school_a")
    assert len(pages) == 1
    assert pages[0]["url"] == "http://x.com/1"


def test_clear_school(db):
    db.enqueue_new(["http://x.com/1"], "school_a")
    db.save_page("http://x.com/1", "school_a", "T", "# Hi", "h1")
    db.clear_school("school_a")
    assert db.next_pending() is None
    assert db.get_pages_for_school("school_a") == []
