import pytest
from src.scraper.cleaner import clean_html


def test_removes_nav_elements():
    html = "<html><nav>Nav links</nav><main><p>Real content</p></main></html>"
    result = clean_html(html, url="http://example.com")
    assert "Nav links" not in result
    assert "Real content" in result


def test_removes_footer():
    html = "<html><footer>Footer stuff</footer><article><p>Article text</p></article></html>"
    result = clean_html(html, url="http://example.com")
    assert "Footer stuff" not in result
    assert "Article text" in result


def test_returns_empty_string_for_blank_page():
    html = "<html><body></body></html>"
    result = clean_html(html, url="http://example.com")
    assert result.strip() == ""


def test_extracts_page_title():
    html = "<html><head><title>Baruch College - Admissions</title></head><body><p>Info</p></body></html>"
    result = clean_html(html, url="http://example.com")
    assert "Baruch College - Admissions" in result


def test_strips_scripts_and_styles():
    html = "<html><body><script>alert('x')</script><style>.a{}</style><p>Text</p></body></html>"
    result = clean_html(html, url="http://example.com")
    assert "alert" not in result
    assert ".a{}" not in result
    assert "Text" in result
