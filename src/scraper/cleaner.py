from bs4 import BeautifulSoup
from markdownify import markdownify as md

REMOVE_TAGS = ["nav", "footer", "header", "script", "style", "aside", "iframe", "form"]


def clean_to_markdown(html: str, url: str) -> str:
    """Extract main content from HTML and convert to Markdown."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.find("body")
    )

    if not main:
        return ""

    markdown = md(str(main), heading_style="ATX", strip=["a"])
    # Collapse excessive blank lines
    lines = markdown.splitlines()
    cleaned = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()
