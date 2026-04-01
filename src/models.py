from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# --- Scraper types ---

@dataclass
class ScrapedPage:
    url: str
    school: str
    text: str        # set to markdown — backward-compatible with ingestion pipeline
    title: str = ""
    page_type: str = "general"
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# --- Generation types ---

@dataclass
class RewrittenQuery:
    query: str
    school: Optional[str]


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]


# --- API types ---

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
