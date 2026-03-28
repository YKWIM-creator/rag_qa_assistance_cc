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
