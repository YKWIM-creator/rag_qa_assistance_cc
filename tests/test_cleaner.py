from src.scraper.cleaner import clean_to_markdown

def test_extracts_heading_as_markdown():
    html = "<html><body><main><h1>Welcome</h1><p>Hello world</p></main></body></html>"
    result = clean_to_markdown(html, url="http://example.com")
    assert "# Welcome" in result
    assert "Hello world" in result

def test_strips_nav_and_footer():
    html = """<html><body>
        <nav>Skip nav</nav>
        <main><p>Main content</p></main>
        <footer>Footer text</footer>
    </body></html>"""
    result = clean_to_markdown(html, url="http://example.com")
    assert "Skip nav" not in result
    assert "Footer text" not in result
    assert "Main content" in result

def test_returns_empty_for_no_content():
    html = "<html><body></body></html>"
    result = clean_to_markdown(html, url="http://example.com")
    assert result == ""

def test_preserves_list_structure():
    html = "<html><body><main><ul><li>Item 1</li><li>Item 2</li></ul></main></body></html>"
    result = clean_to_markdown(html, url="http://example.com")
    assert "Item 1" in result
    assert "Item 2" in result
