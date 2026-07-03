import os
from pathlib import Path
from typing import Literal

from pydantic import Field , field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

"""
config/settings.py
Production-grade application settings using Pydantic v2 BaseSettings.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration validated at startup.
    Loads from .env file if present; falls back to sensible defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Identity
    app_name: str = Field(default="Enterprise Semantic RAG")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # File System Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    vector_store_path: Path = Field(default=Path("data/vector_store"))

    # Ingestion / Chunking
    chunk_size: int = Field(default=512, ge=128, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=1024)

    # Embedding Model
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = Field(default="auto")

    # Retrieval & Re-ranking
    top_k_dense: int = Field(default=10, ge=1, le=100)
    top_k_sparse: int = Field(default=10, ge=1, le=100)
    top_k_final: int = Field(default=5, ge=1, le=50)
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    cross_encoder_batch_size: int = Field(default=32, ge=1, le=256)

    # LLM
    llm_provider: Literal["openai", "ollama"] = Field(default="openai")
    llm_model_name: str = Field(default="gpt-4o-mini")
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1, le=4096)

    # FastAPI
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        return v

    @field_validator("vector_store_path", "raw_data_dir", "data_dir")
    @classmethod
    def resolve_paths(cls, v: Path) -> Path:
        return v.resolve()


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
