import json
import logging

from src.models import RewrittenQuery

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """You are a query preprocessing assistant for a CUNY university information system.

Given a user question, return a JSON object with:
- "school": the CUNY school short name if the question targets a specific school, or null if general.
  Valid school names: baruch, brooklyn, city, hunter, john_jay, lehman, medgar_evers, nycct, queens, staten_island, york
- "query": a rewritten, retrieval-optimized version of the question (more specific, academic language)

Question: {question}

Return ONLY valid JSON. Example: {{"school": "baruch", "query": "Baruch College undergraduate admissions requirements GPA"}}"""


def rewrite_query(question: str, llm) -> RewrittenQuery:
    """Extract school + rewrite question for better retrieval. Falls back gracefully."""
    prompt = _REWRITE_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        data = json.loads(content)
        return RewrittenQuery(
            query=data.get("query", question),
            school=data.get("school") or None,
        )
    except Exception as e:
        logger.warning(f"Query rewriting failed, using original: {e}")
        return RewrittenQuery(query=question, school=None)
