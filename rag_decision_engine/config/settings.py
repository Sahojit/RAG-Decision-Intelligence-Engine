import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    app_name: str = "RAG Decision Intelligence Engine"
    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_timeout: int = 120
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_decision_engine"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    @property
    def postgres_dsn(self) -> str:
        auth = self.postgres_user
        if self.postgres_password:
            auth = f"{self.postgres_user}:{self.postgres_password}"
        return f"postgresql://{auth}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    faiss_index_path: Path = DATA_DIR / "embeddings" / "faiss.index"
    faiss_metadata_path: Path = DATA_DIR / "embeddings" / "metadata.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    contradiction_threshold: float = 0.5
    ml_model_type: Literal["xgboost", "random_forest"] = "xgboost"
    ml_model_path: Path = ROOT_DIR / "models" / "reliability_model.pkl"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "evidence-reliability"
    source_credibility_map: dict[str, float] = Field(
        default={
            "research_paper": 0.95,
            "official_documentation": 0.90,
            "technical_blog": 0.70,
            "news_article": 0.60,
            "forum": 0.40,
            "unknown": 0.50,
        }
    )
    weight_similarity: float = 0.5
    weight_reliability: float = 0.3
    weight_credibility: float = 0.2
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    logs_dir: Path = ROOT_DIR / "logs"
settings = Settings()
