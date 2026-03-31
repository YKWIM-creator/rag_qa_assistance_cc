from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from src.generation.rewriter import rewrite_query


PROMPT_TEMPLATE = """You are a CUNY student assistant. Answer the question using ONLY the context below.
Each context block is labeled with [School | page type | section].
If the question asks about a specific school, prioritize blocks from that school.
If the answer is not found in the context, say: "I don't have information about that in the CUNY documents I've indexed."
Be concise and cite the school name in your answer.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]


def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        school = doc.metadata.get("school", "unknown").title()
        page_type = doc.metadata.get("page_type", "general")
        heading = doc.metadata.get("section_heading", doc.metadata.get("title", ""))
        label = f"[{school} | {page_type} | {heading}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


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
    """Run RAG pipeline: rewrite query → retrieve → generate → return answer + sources."""
    # Rewrite query and extract school
    rewritten = rewrite_query(question, llm)

    # Apply or clear school filter
    if hasattr(retriever, "search_kwargs"):
        if rewritten.school:
            retriever.search_kwargs["filter"] = {"school": rewritten.school}
        else:
            retriever.search_kwargs.pop("filter", None)

    docs = retriever.invoke(rewritten.query)

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
