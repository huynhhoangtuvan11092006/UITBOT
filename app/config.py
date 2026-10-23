"""Application settings loaded from environment and optional .env file."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "uitbot"
    postgres_user: str = "uitbot_user"
    postgres_password: str = "uitbot_password"

    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:20128/v1"
    openai_model: str = ""

    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = False

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_batch_size: int = 16

    chunk_strategy: Literal["fixed", "sentence", "structured"] = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 150
    chunk_sentence_overlap: int = 1

    ocr_engine: Literal["tesseract", "none"] = "tesseract"
    ocr_min_chars: int = 50
    ocr_replacement_char_ratio: float = 0.02
    ocr_min_vietnamese_ratio: float = 0.15
    ocr_low_confidence: float = 60.0
    ocr_dpi: int = 300

    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True
    reranker_batch_size: int = 16
    retrieval_candidate_count: int = 20
    rrf_k: int = 60
    bm25_tokenizer: Literal["underthesea", "simple"] = "underthesea"

    evidence_min_score: float = 0.02
    rag_top_k: int = 5
    llm_temperature: float = 0.1
    llm_timeout: float = 45.0

    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")

    def postgres_conninfo(self) -> str:
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


settings = Settings()
