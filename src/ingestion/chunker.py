from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.scraper.spider import ScrapedPage

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,       # characters (~500 tokens)
    chunk_overlap=200,     # characters (~50 tokens)
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_page(page: ScrapedPage) -> list[dict]:
    """Split a ScrapedPage into chunks with metadata."""
    if not page.text.strip():
        return []

    texts = splitter.split_text(page.text)
    chunks = []
    for i, text in enumerate(texts):
        chunks.append({
            "text": text,
            "metadata": {
                "url": page.url,
                "school": page.school,
                "title": page.title,
                "chunk_index": i,
                "scraped_at": page.scraped_at,
            }
        })
    return chunks
