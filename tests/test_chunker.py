import pytest
from src.ingestion.chunker import chunk_page
from src.models import ScrapedPage


def make_page(text: str, url: str = "http://test.com", school: str = "test",
              title: str = "Test Page", page_type: str = "general") -> ScrapedPage:
    return ScrapedPage(url=url, school=school, text=text, title=title, page_type=page_type)


def test_short_text_produces_one_chunk():
    page = make_page("This is a short text.")
    chunks = chunk_page(page)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks():
    long_text = "Word " * 600
    page = make_page(long_text)
    chunks = chunk_page(page)
    assert len(chunks) > 1


def test_chunks_contain_base_metadata():
    page = make_page("Some content.", url="http://baruch.cuny.edu/admissions",
                     school="baruch", page_type="admissions")
    chunks = chunk_page(page)
    meta = chunks[0]["metadata"]
    assert meta["url"] == "http://baruch.cuny.edu/admissions"
    assert meta["school"] == "baruch"
    assert meta["page_type"] == "admissions"
    assert "chunk_index" in meta


def test_chunks_contain_section_heading():
    md = "# Welcome\n\n## Admissions Requirements\n\nYou must submit transcripts.\n"
    page = make_page(md, title="Admissions | Baruch")
    chunks = chunk_page(page)
    headings = [c["metadata"]["section_heading"] for c in chunks]
    assert any("Admissions Requirements" in h for h in headings)


def test_section_heading_falls_back_to_title():
    page = make_page("No headings here, just plain text.", title="About Baruch")
    chunks = chunk_page(page)
    assert chunks[0]["metadata"]["section_heading"] == "About Baruch"


def test_chunks_have_text_field():
    page = make_page("Hello world content.")
    chunks = chunk_page(page)
    assert "text" in chunks[0]
    assert len(chunks[0]["text"]) > 0


def test_empty_text_returns_no_chunks():
    page = make_page("")
    chunks = chunk_page(page)
    assert chunks == []


def test_markdown_header_split_keeps_section_together():
    md = "## Requirements\n\n" + "requirement detail " * 20 + "\n\n## Deadlines\n\n" + "deadline info " * 20
    page = make_page(md)
    chunks = chunk_page(page)
    # No chunk should contain content from both sections
    for chunk in chunks:
        text = chunk["text"]
        has_req = "requirement detail" in text
        has_dead = "deadline info" in text
        assert not (has_req and has_dead), f"Chunk spans two sections: {text[:80]}"


def test_bare_heading_produces_no_empty_chunks():
    md = "## Empty Section\n"
    page = make_page(md, title="Test")
    chunks = chunk_page(page)
    for chunk in chunks:
        assert chunk["text"].strip() != "", "Got a chunk with empty text"
