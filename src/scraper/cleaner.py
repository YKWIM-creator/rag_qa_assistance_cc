from bs4 import BeautifulSoup


REMOVE_TAGS = ["nav", "footer", "header", "script", "style", "aside", "iframe", "form"]


def clean_html(html: str, url: str) -> str:
    """Extract main text content from raw HTML, stripping navigation/boilerplate."""
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate tags
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    # Try to find main content area
    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else ""

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.find("body")
    )

    if not main:
        return ""

    text = main.get_text(separator="\n", strip=True)

    if title_text:
        text = f"{title_text}\n\n{text}"

    return text
