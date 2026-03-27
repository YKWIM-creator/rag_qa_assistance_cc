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
