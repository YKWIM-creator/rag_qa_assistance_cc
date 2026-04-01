from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.models import ScrapedPage

_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_page(page: ScrapedPage) -> list[dict]:
    """Split a ScrapedPage into chunks with enriched metadata."""
    if not page.text.strip():
        return []

    # Pass 1: split on Markdown headers
    header_docs = _HEADER_SPLITTER.split_text(page.text)

    # Pass 2: character-split any section that's still too large
    final_chunks = []
    for doc in header_docs:
        sub_texts = _CHAR_SPLITTER.split_text(doc.page_content)
        for sub in sub_texts:
            if not sub.strip():
                continue
            final_chunks.append((sub, doc.metadata))

    chunks = []
    for i, (text, header_meta) in enumerate(final_chunks):
        # Build section_heading: prefer deepest header found, fallback to page title
        section_heading = (
            header_meta.get("h3")
            or header_meta.get("h2")
            or header_meta.get("h1")
            or page.title
        )
        chunks.append({
            "text": text,
            "metadata": {
                "url": page.url,
                "school": page.school,
                "title": page.title,
                "section_heading": section_heading,
                "page_type": page.page_type,
                "chunk_index": i,
                "scraped_at": page.scraped_at,
            },
        })
    return chunks
