import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.generation.chain import ask
from src.generation.providers import get_llm
from src.ingestion.embedder import get_embedding_model
from src.models import QuestionRequest, AnswerResponse, RAGResponse
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        result: RAGResponse = ask(request.question, retriever, llm)
        return AnswerResponse(answer=result.answer, sources=result.sources)
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=503, detail="Failed to generate answer")


@app.get("/sources")
def list_sources():
    """Return indexed schools."""
    return {"schools": list(settings.cuny_senior_colleges.keys())}
