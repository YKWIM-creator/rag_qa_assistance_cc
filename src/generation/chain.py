from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.generation.rewriter import rewrite_query
from src.models import RAGResponse
from src.retrieval.retriever import get_retriever


PROMPT_TEMPLATE = """You are a CUNY student assistant. Answer the question using ONLY the context below.
Each context block is labeled with [School | page type | section].
If the question asks about a specific school, prioritize blocks from that school.
If the answer is not found in the context, say: "I don't have information about that in the CUNY documents I've indexed."
Be concise and cite the school name in your answer.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        school = doc.metadata.get("school", "unknown").title()
        page_type = doc.metadata.get("page_type", "general")
        heading = doc.metadata.get("section_heading", doc.metadata.get("title", ""))
        label = f"[{school} | {page_type} | {heading}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, retriever, llm) -> RAGResponse:
    """Run RAG pipeline: rewrite query → retrieve → generate → return answer + sources."""
    # Rewrite query and extract school
    rewritten = rewrite_query(question, llm)

    # Create a scoped retriever per call to avoid shared-state mutation across concurrent requests
    active_retriever = retriever
    if hasattr(retriever, "vectorstore"):
        metadata_filter = {"school": rewritten.school} if rewritten.school else None
        active_retriever = get_retriever(retriever.vectorstore, metadata_filter=metadata_filter)

    docs = active_retriever.invoke(rewritten.query)

    if not docs:
        return RAGResponse(
            answer="I don't have information about that in the CUNY documents I've indexed.",
            sources=[],
        )

    # Retrieve once and build a simple chain with pre-formatted context
    context = _format_docs(docs)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": rewritten.query})

    sources = [
        {
            "url": doc.metadata.get("url", ""),
            "school": doc.metadata.get("school", ""),
            "title": doc.metadata.get("title", ""),
        }
        for doc in docs
    ]
    seen_urls: set[str] = set()
    unique_sources = []
    for s in sources:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            unique_sources.append(s)

    return RAGResponse(answer=answer, sources=unique_sources)
