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
    scraper_db_path: str = Field(default="./scraper_cache/scraper.db", env="SCRAPER_DB_PATH")

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
