# CUNY RAG Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end RAG assistant that answers student questions grounded in scraped CUNY documentation.

**Architecture:** LangChain orchestrates the full pipeline — async web scraper feeds cleaned text into ChromaDB via LangChain embeddings, a RAG chain retrieves top-k chunks with MMR and generates grounded answers via a swappable LLM provider, exposed through FastAPI and a Streamlit chat UI, evaluated with RAGAS.

**Tech Stack:** Python 3.11+, LangChain, ChromaDB, httpx, BeautifulSoup4, FastAPI, Uvicorn, Streamlit, RAGAS, pytest, python-dotenv

---

## Task 1: Project Scaffold & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config/settings.py`
- Create: `config/__init__.py`
- Create: `src/__init__.py`
- Create: `src/scraper/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/retrieval/__init__.py`
- Create: `src/generation/__init__.py`
- Create: `src/api/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `vectorstore/.gitkeep`

**Step 1: Create requirements.txt**

```
httpx==0.27.0
beautifulsoup4==4.12.3
lxml==5.2.2
langchain==0.2.16
langchain-openai==0.1.23
langchain-anthropic==0.1.23
langchain-community==0.2.16
chromadb==0.5.5
fastapi==0.111.1
uvicorn==0.30.3
streamlit==1.36.0
ragas==0.1.14
python-dotenv==1.0.1
pydantic==2.8.2
pydantic-settings==2.3.4
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-mock==3.14.0
tenacity==8.5.0
```

**Step 2: Create .env.example**

```
# LLM Provider - set ONE of these
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# For Ollama (local), no key needed

# Active provider: openai | anthropic | ollama
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Ollama (if using local)
OLLAMA_BASE_URL=http://localhost:11434

# ChromaDB
CHROMA_PERSIST_DIR=./vectorstore
CHROMA_COLLECTION_NAME=cuny_docs

# Scraper
SCRAPER_RATE_LIMIT_DELAY=1.0
SCRAPER_MAX_RETRIES=3
SCRAPER_TIMEOUT=10

# API
API_HOST=0.0.0.0
API_PORT=8000
```

**Step 3: Create config/settings.py**

```python
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o", env="LLM_MODEL")
    embedding_provider: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./vectorstore", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="cuny_docs", env="CHROMA_COLLECTION_NAME")

    # Scraper
    scraper_rate_limit_delay: float = Field(default=1.0, env="SCRAPER_RATE_LIMIT_DELAY")
    scraper_max_retries: int = Field(default=3, env="SCRAPER_MAX_RETRIES")
    scraper_timeout: int = Field(default=10, env="SCRAPER_TIMEOUT")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")

    # CUNY senior colleges
    cuny_senior_colleges: dict = {
        "baruch": "https://www.baruch.cuny.edu",
        "brooklyn": "https://www.brooklyn.cuny.edu",
        "city": "https://www.ccny.cuny.edu",
        "hunter": "https://www.hunter.cuny.edu",
        "john_jay": "https://www.jjay.cuny.edu",
        "lehman": "https://www.lehman.cuny.edu",
        "medgar_evers": "https://www.mec.cuny.edu",
        "nycct": "https://www.citytech.cuny.edu",
        "queens": "https://www.qc.cuny.edu",
        "staten_island": "https://www.csi.cuny.edu",
        "york": "https://www.york.cuny.edu",
    }

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
```

**Step 4: Create all __init__.py and .gitkeep files**

```bash
touch config/__init__.py src/__init__.py src/scraper/__init__.py \
  src/ingestion/__init__.py src/retrieval/__init__.py \
  src/generation/__init__.py src/api/__init__.py \
  src/evaluation/__init__.py tests/__init__.py \
  data/raw/.gitkeep data/processed/.gitkeep vectorstore/.gitkeep
```

**Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

**Step 6: Copy .env.example to .env and fill in your API key**

```bash
cp .env.example .env
# Then edit .env and set your OPENAI_API_KEY (or other provider)
```

**Step 7: Verify settings load**

```bash
python -c "from config.settings import settings; print(settings.llm_provider)"
```

Expected output: `openai`

**Step 8: Commit**

```bash
git add requirements.txt .env.example config/ src/ tests/ data/ vectorstore/
git commit -m "feat: project scaffold, settings, and dependencies"
```

---

## Task 2: HTML Cleaner

**Files:**
- Create: `src/scraper/cleaner.py`
- Create: `tests/test_cleaner.py`

**Step 1: Write the failing tests**

```python
# tests/test_cleaner.py
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cleaner.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `clean_html` does not exist yet.

**Step 3: Implement cleaner**

```python
# src/scraper/cleaner.py
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cleaner.py -v
```

Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add src/scraper/cleaner.py tests/test_cleaner.py
git commit -m "feat: HTML cleaner with boilerplate stripping"
```

---

## Task 3: Async Web Scraper

**Files:**
- Create: `src/scraper/spider.py`
- Create: `tests/test_spider.py`

**Step 1: Write the failing tests**

```python
# tests/test_spider.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.scraper.spider import scrape_page, ScrapedPage


@pytest.mark.asyncio
async def test_scrape_page_returns_scraped_page_on_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><head><title>Test</title></head><body><main><p>Hello</p></main></body></html>"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await scrape_page("http://example.com", school="test")

    assert isinstance(result, ScrapedPage)
    assert result.url == "http://example.com"
    assert result.school == "test"
    assert "Hello" in result.text


@pytest.mark.asyncio
async def test_scrape_page_returns_none_on_404():
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await scrape_page("http://example.com/notfound", school="test")

    assert result is None


def test_scraped_page_has_required_fields():
    page = ScrapedPage(url="http://x.com", school="baruch", text="text", title="Title")
    assert page.url == "http://x.com"
    assert page.school == "baruch"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_spider.py -v
```

Expected: `ImportError` — spider module does not exist.

**Step 3: Implement spider**

```python
# src/scraper/spider.py
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.scraper.cleaner import clean_html

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    url: str
    school: str
    text: str
    title: str = ""
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@retry(
    stop=stop_after_attempt(settings.scraper_max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=False,
)
async def scrape_page(url: str, school: str) -> Optional[ScrapedPage]:
    """Fetch a single URL and return a ScrapedPage, or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=settings.scraper_timeout, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.warning(f"Skipping {url}: HTTP {response.status_code}")
                return None

            text = clean_html(response.text, url=url)
            if not text.strip():
                return None

            soup = BeautifulSoup(response.text, "lxml")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            return ScrapedPage(url=url, school=school, text=text, title=title)

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all internal links from a page."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only same-domain, http/https, no fragments
        if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
            links.append(full_url.split("#")[0])  # strip fragments

    return list(set(links))


async def crawl_school(school: str, start_url: str, max_pages: int = 500) -> list[ScrapedPage]:
    """BFS crawl of a single CUNY school site."""
    seen: set[str] = set()
    queue: list[str] = [start_url]
    results: list[ScrapedPage] = []

    async with httpx.AsyncClient(timeout=settings.scraper_timeout, follow_redirects=True) as client:
        while queue and len(results) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            await asyncio.sleep(settings.scraper_rate_limit_delay)

            try:
                response = await client.get(url)
                if response.status_code != 200:
                    continue

                page = await scrape_page(url, school)
                if page:
                    results.append(page)
                    logger.info(f"[{school}] Scraped: {url}")

                new_links = extract_links(response.text, url)
                for link in new_links:
                    if link not in seen:
                        queue.append(link)

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                continue

    logger.info(f"[{school}] Done: {len(results)} pages")
    return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_spider.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/scraper/spider.py tests/test_spider.py
git commit -m "feat: async web spider with BFS crawl and retry"
```

---

## Task 4: Chunker

**Files:**
- Create: `src/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunker.py
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ImportError` — chunker does not exist.

**Step 3: Implement chunker**

```python
# src/ingestion/chunker.py
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: all 5 tests PASS.

**Step 5: Commit**

```bash
git add src/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: LangChain text chunker with metadata"
```

---

## Task 5: Embedding Provider Abstraction

**Files:**
- Create: `src/ingestion/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing tests**

```python
# tests/test_embedder.py
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.embedder import get_embedding_model


def test_get_embedding_model_openai():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_provider = "openai"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.openai_api_key = "sk-test"
        model = get_embedding_model()
    assert model is not None
    assert "OpenAI" in type(model).__name__


def test_get_embedding_model_anthropic():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_provider = "anthropic"
        mock_settings.embedding_model = "voyage-3"
        mock_settings.anthropic_api_key = "sk-ant-test"
        model = get_embedding_model()
    assert model is not None


def test_get_embedding_model_unknown_raises():
    with patch("src.ingestion.embedder.settings") as mock_settings:
        mock_settings.embedding_provider = "unknown_provider"
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_model()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_embedder.py -v
```

Expected: `ImportError` — embedder does not exist.

**Step 3: Implement embedder**

```python
# src/ingestion/embedder.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from config.settings import settings


def get_embedding_model():
    """Return a LangChain embedding model based on configured provider."""
    provider = settings.embedding_provider

    if provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    elif provider == "anthropic":
        # Anthropic uses VoyageAI for embeddings via langchain-community
        try:
            from langchain_community.embeddings import VoyageEmbeddings
            return VoyageEmbeddings(
                voyage_api_key=settings.anthropic_api_key,
                model=settings.embedding_model,
            )
        except ImportError:
            raise ImportError("Install langchain-community for Anthropic/Voyage embeddings")
    elif provider == "ollama":
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use openai, anthropic, or ollama.")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/ingestion/embedder.py tests/test_embedder.py
git commit -m "feat: embedding provider abstraction (OpenAI/Anthropic/Ollama)"
```

---

## Task 6: ChromaDB Vector Store Integration

**Files:**
- Create: `src/retrieval/retriever.py`
- Create: `tests/test_retriever.py`

**Step 1: Write the failing tests**

```python
# tests/test_retriever.py
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from src.retrieval.retriever import build_vectorstore, get_retriever


def test_build_vectorstore_returns_chroma_instance():
    chunks = [
        {"text": "Baruch offers a BBA program.", "metadata": {"url": "http://baruch.cuny.edu", "school": "baruch", "title": "Programs", "chunk_index": 0, "scraped_at": "2026-01-01"}},
        {"text": "Hunter has a nursing program.", "metadata": {"url": "http://hunter.cuny.edu", "school": "hunter", "title": "Programs", "chunk_index": 0, "scraped_at": "2026-01-01"}},
    ]
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents = MagicMock(return_value=[[0.1] * 384, [0.2] * 384])
    mock_embeddings.embed_query = MagicMock(return_value=[0.15] * 384)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.retrieval.retriever.settings") as mock_settings:
            mock_settings.chroma_persist_dir = tmpdir
            mock_settings.chroma_collection_name = "test_collection"
            vs = build_vectorstore(chunks, mock_embeddings)

    assert vs is not None


def test_get_retriever_returns_langchain_retriever():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever = MagicMock(return_value=MagicMock())
    retriever = get_retriever(mock_vectorstore, k=3)
    mock_vectorstore.as_retriever.assert_called_once()
    assert retriever is not None
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retriever.py -v
```

Expected: `ImportError` — retriever does not exist.

**Step 3: Implement retriever**

```python
# src/retrieval/retriever.py
import logging
from langchain_community.vectorstores import Chroma
from config.settings import settings

logger = logging.getLogger(__name__)


def build_vectorstore(chunks: list[dict], embedding_model) -> Chroma:
    """Build (or load) a ChromaDB vector store from chunks."""
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{c['metadata']['url']}__chunk_{c['metadata']['chunk_index']}" for c in chunks]

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
    )

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta, ids=batch_ids)
        logger.info(f"Indexed batch {i // batch_size + 1} ({len(batch_texts)} chunks)")

    return vectorstore


def load_vectorstore(embedding_model) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(vectorstore: Chroma, k: int = 5):
    """Return a LangChain MMR retriever from a ChromaDB vector store."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Expected: all 2 tests PASS.

**Step 5: Commit**

```bash
git add src/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: ChromaDB vector store with MMR retriever"
```

---

## Task 7: LLM Provider Abstraction + RAG Chain

**Files:**
- Create: `src/generation/providers.py`
- Create: `src/generation/chain.py`
- Create: `tests/test_chain.py`

**Step 1: Write the failing tests**

```python
# tests/test_chain.py
import pytest
from unittest.mock import patch, MagicMock
from src.generation.providers import get_llm
from src.generation.chain import build_rag_chain, RAGResponse


def test_get_llm_openai():
    with patch("src.generation.providers.settings") as mock_settings:
        mock_settings.llm_provider = "openai"
        mock_settings.llm_model = "gpt-4o"
        mock_settings.openai_api_key = "sk-test"
        llm = get_llm()
    assert llm is not None
    assert "OpenAI" in type(llm).__name__


def test_get_llm_unknown_raises():
    with patch("src.generation.providers.settings") as mock_settings:
        mock_settings.llm_provider = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm()


def test_build_rag_chain_returns_chain():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    chain = build_rag_chain(mock_retriever, mock_llm)
    assert chain is not None


def test_rag_response_has_required_fields():
    resp = RAGResponse(answer="42", sources=[{"url": "http://x.com", "school": "baruch", "title": "Test"}])
    assert resp.answer == "42"
    assert len(resp.sources) == 1
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chain.py -v
```

Expected: `ImportError`.

**Step 3: Implement providers.py**

```python
# src/generation/providers.py
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config.settings import settings


def get_llm():
    """Return a LangChain LLM based on configured provider."""
    provider = settings.llm_provider

    if provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use openai, anthropic, or ollama.")
```

**Step 4: Implement chain.py**

```python
# src/generation/chain.py
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


PROMPT_TEMPLATE = """You are a CUNY student assistant. Answer the question using ONLY the context below.
If the answer is not found in the context, say: "I don't have information about that in the CUNY documents I've indexed."
Be concise and helpful.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm):
    """Build a LangChain RAG chain from a retriever and LLM."""
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, retriever, llm) -> RAGResponse:
    """Run RAG pipeline for a question, return answer + sources."""
    # Get docs for sources
    docs = retriever.invoke(question)

    if not docs:
        return RAGResponse(
            answer="I don't have information about that in the CUNY documents I've indexed.",
            sources=[],
        )

    chain = build_rag_chain(retriever, llm)
    answer = chain.invoke(question)

    sources = [
        {
            "url": doc.metadata.get("url", ""),
            "school": doc.metadata.get("school", ""),
            "title": doc.metadata.get("title", ""),
        }
        for doc in docs
    ]
    # Deduplicate sources by URL
    seen_urls = set()
    unique_sources = []
    for s in sources:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            unique_sources.append(s)

    return RAGResponse(answer=answer, sources=unique_sources)
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_chain.py -v
```

Expected: all 4 tests PASS.

**Step 6: Commit**

```bash
git add src/generation/providers.py src/generation/chain.py tests/test_chain.py
git commit -m "feat: LLM provider abstraction and RAG chain"
```

---

## Task 8: Ingestion Pipeline Script

**Files:**
- Create: `src/ingestion/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.ingestion.pipeline import ingest_pages


@pytest.mark.asyncio
async def test_ingest_pages_stores_chunks():
    from src.scraper.spider import ScrapedPage

    pages = [
        ScrapedPage(url="http://baruch.cuny.edu", school="baruch", text="Baruch offers business programs.", title="Home"),
    ]

    mock_embeddings = MagicMock()
    mock_vectorstore = MagicMock()

    with patch("src.ingestion.pipeline.build_vectorstore", return_value=mock_vectorstore) as mock_build:
        result = await ingest_pages(pages, mock_embeddings)

    mock_build.assert_called_once()
    assert result == mock_vectorstore
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ImportError`.

**Step 3: Implement pipeline**

```python
# src/ingestion/pipeline.py
import logging
from src.scraper.spider import ScrapedPage
from src.ingestion.chunker import chunk_page
from src.retrieval.retriever import build_vectorstore

logger = logging.getLogger(__name__)


async def ingest_pages(pages: list[ScrapedPage], embedding_model):
    """Chunk all scraped pages and store in ChromaDB."""
    all_chunks = []
    for page in pages:
        chunks = chunk_page(page)
        all_chunks.extend(chunks)
        logger.info(f"Chunked {page.url}: {len(chunks)} chunks")

    logger.info(f"Total chunks to index: {len(all_chunks)}")
    vectorstore = build_vectorstore(all_chunks, embedding_model)
    return vectorstore
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingestion pipeline - chunk and index scraped pages"
```

---

## Task 9: FastAPI Backend

**Files:**
- Create: `src/api/main.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing tests**

```python
# tests/test_api.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_health_endpoint():
    with patch("src.api.main.retriever", MagicMock()), \
         patch("src.api.main.llm", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_returns_answer():
    mock_rag_response = MagicMock()
    mock_rag_response.answer = "Baruch is located in Manhattan."
    mock_rag_response.sources = [{"url": "http://baruch.cuny.edu", "school": "baruch", "title": "About"}]

    with patch("src.api.main.ask", return_value=mock_rag_response), \
         patch("src.api.main.retriever", MagicMock()), \
         patch("src.api.main.llm", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": "Where is Baruch?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_ask_endpoint_rejects_empty_question():
    with patch("src.api.main.retriever", MagicMock()), \
         patch("src.api.main.llm", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError`.

**Step 3: Implement FastAPI app**

```python
# src/api/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.generation.chain import ask as rag_ask, RAGResponse
from src.generation.providers import get_llm
from src.ingestion.embedder import get_embedding_model
from src.retrieval.retriever import load_vectorstore, get_retriever

logger = logging.getLogger(__name__)

retriever = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm
    logger.info("Loading vector store and LLM...")
    embedding_model = get_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    logger.info("Ready.")
    yield


app = FastAPI(title="CUNY RAG Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        result: RAGResponse = rag_ask(request.question, retriever, llm)
        return AnswerResponse(answer=result.answer, sources=result.sources)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=503, detail="Failed to generate answer")


@app.get("/sources")
def list_sources():
    """Return indexed schools."""
    return {"schools": list(settings.cuny_senior_colleges.keys())}
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add src/api/main.py tests/test_api.py
git commit -m "feat: FastAPI backend with /ask, /health, /sources endpoints"
```

---

## Task 10: Streamlit Chat UI

**Files:**
- Create: `ui/app.py`

**Note:** Streamlit apps are not unit-testable in the traditional sense. Manual verification is used here.

**Step 1: Implement Streamlit UI**

```python
# ui/app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CUNY Assistant", page_icon="🎓", layout="centered")
st.title("🎓 CUNY Student Assistant")
st.caption("Ask questions about CUNY programs, admissions, financial aid, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(f"- [{s.get('title', s['url'])}]({s['url']}) — *{s.get('school', '')}*")

if prompt := st.chat_input("Ask a question about CUNY..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching CUNY documents..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": prompt}, timeout=30)
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except requests.exceptions.ConnectionError:
                answer = "Could not connect to the CUNY assistant API. Make sure the backend is running."
                sources = []
            except Exception as e:
                answer = f"An error occurred: {str(e)}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s.get('title', s['url'])}]({s['url']}) — *{s.get('school', '')}*")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
```

**Step 2: Manually verify the UI**

Start the API in one terminal:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Start Streamlit in another:
```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser. Verify:
- Chat input appears
- Submitting a question calls the API
- Answer and sources are displayed

**Step 3: Commit**

```bash
git add ui/app.py
git commit -m "feat: Streamlit chat UI with source citations"
```

---

## Task 11: RAGAS Evaluation

**Files:**
- Create: `src/evaluation/eval.py`
- Create: `data/golden_dataset.json`
- Create: `tests/test_eval.py`

**Step 1: Create a small golden dataset**

```json
[
  {
    "question": "What majors does Baruch College offer?",
    "ground_truth": "Baruch College offers majors in business, liberal arts, and sciences through its three schools."
  },
  {
    "question": "Where is Hunter College located?",
    "ground_truth": "Hunter College is located in Manhattan, New York City."
  },
  {
    "question": "Does CUNY offer financial aid?",
    "ground_truth": "Yes, CUNY offers financial aid including federal, state, and institutional aid programs."
  }
]
```

Save this as `data/golden_dataset.json`.

**Step 2: Write the failing test**

```python
# tests/test_eval.py
import pytest
import json
from src.evaluation.eval import load_golden_dataset, format_ragas_dataset


def test_load_golden_dataset():
    dataset = load_golden_dataset("data/golden_dataset.json")
    assert len(dataset) > 0
    assert "question" in dataset[0]
    assert "ground_truth" in dataset[0]


def test_format_ragas_dataset():
    samples = [
        {
            "question": "What is CUNY?",
            "ground_truth": "CUNY is the City University of New York.",
            "answer": "CUNY stands for City University of New York.",
            "contexts": ["CUNY is the City University of New York, a public university system."],
        }
    ]
    dataset = format_ragas_dataset(samples)
    assert dataset is not None
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_eval.py -v
```

Expected: `ImportError`.

**Step 4: Implement eval.py**

```python
# src/evaluation/eval.py
import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

logger = logging.getLogger(__name__)


def load_golden_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def format_ragas_dataset(samples: list[dict]) -> Dataset:
    """Convert list of QA dicts to a HuggingFace Dataset for RAGAS."""
    return Dataset.from_list(samples)


def run_evaluation(retriever, llm, golden_path: str = "data/golden_dataset.json") -> dict:
    """Run RAGAS evaluation on the golden dataset."""
    from src.generation.chain import ask

    golden = load_golden_dataset(golden_path)
    samples = []

    for item in golden:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Get RAG response
        response = ask(question, retriever, llm)
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        samples.append({
            "question": question,
            "answer": response.answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    dataset = format_ragas_dataset(samples)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    logger.info(f"RAGAS results: {results}")
    return results
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_eval.py -v
```

Expected: all 2 tests PASS.

**Step 6: Commit**

```bash
git add src/evaluation/eval.py data/golden_dataset.json tests/test_eval.py
git commit -m "feat: RAGAS evaluation pipeline with golden dataset"
```

---

## Task 12: Scrape Runner Script

**Files:**
- Create: `scripts/run_scrape.py`

**Step 1: Implement the runner**

```python
# scripts/run_scrape.py
"""
Run the full CUNY scrape + ingestion pipeline.

Usage:
    python scripts/run_scrape.py                    # scrape all schools
    python scripts/run_scrape.py --school baruch    # scrape one school
    python scripts/run_scrape.py --max-pages 100    # limit pages per school
"""
import asyncio
import argparse
import logging
import sys

sys.path.insert(0, ".")

from config.settings import settings
from src.scraper.spider import crawl_school
from src.ingestion.pipeline import ingest_pages
from src.ingestion.embedder import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(school_filter: str | None, max_pages: int):
    embedding_model = get_embedding_model()
    schools = settings.cuny_senior_colleges

    if school_filter:
        if school_filter not in schools:
            logger.error(f"Unknown school: {school_filter}. Options: {list(schools.keys())}")
            return
        schools = {school_filter: schools[school_filter]}

    all_pages = []
    for school, url in schools.items():
        logger.info(f"Crawling {school}: {url}")
        pages = await crawl_school(school, url, max_pages=max_pages)
        all_pages.extend(pages)
        logger.info(f"Collected {len(pages)} pages from {school}")

    logger.info(f"Total pages: {len(all_pages)}. Starting ingestion...")
    await ingest_pages(all_pages, embedding_model)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", type=str, default=None)
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(main(args.school, args.max_pages))
```

**Step 2: Test the runner with a single school, limited pages**

```bash
python scripts/run_scrape.py --school baruch --max-pages 10
```

Expected: logs showing pages scraped and indexed, no errors.

**Step 3: Commit**

```bash
git add scripts/run_scrape.py
git commit -m "feat: scrape runner script with per-school and max-pages options"
```

---

## Task 13: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS.

**Step 2: Run a real end-to-end smoke test**

Make sure your `.env` has a valid API key and you've run the scraper for at least one school. Then:

```bash
# Terminal 1: start API
uvicorn src.api.main:app --reload

# Terminal 2: ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What programs does Baruch College offer?"}'
```

Expected: JSON response with `answer` and `sources`.

**Step 3: Commit**

```bash
git add .
git commit -m "chore: all tests passing, end-to-end verified"
```

---

## Verification Checklist

- [ ] `pytest tests/ -v` — all tests pass
- [ ] `python scripts/run_scrape.py --school baruch --max-pages 20` — scrapes and indexes without error
- [ ] `uvicorn src.api.main:app` — API starts and `/health` returns `{"status": "ok"}`
- [ ] `/ask` endpoint returns a grounded answer with sources
- [ ] `streamlit run ui/app.py` — chat UI loads and returns answers
- [ ] RAGAS eval runs: `python -c "from src.evaluation.eval import run_evaluation; print('eval ready')"`
