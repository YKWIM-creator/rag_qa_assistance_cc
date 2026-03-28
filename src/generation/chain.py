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
