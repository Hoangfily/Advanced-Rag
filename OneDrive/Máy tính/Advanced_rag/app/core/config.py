from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Advanced RAG"

    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "documents"

    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-flash-lite-latest"

    chunk_size: int = 512
    chunk_overlap: int = 64

    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5

    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    enable_query_expansion: bool = False
    query_expansion_variants: int = 2

    gemini_api_key: str = ""

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
