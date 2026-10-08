"""Configuration management using Pydantic Settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_env: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )
    app_name: str = Field(default="semantic-search-engine", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")

    # API Server
    api_host: str = Field(default="0.0.0.0", description="API host")  # nosec B104
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="API workers")

    # Security
    api_key_admin: str = Field(default="admin-secret-key-12345", description="Admin API key")
    api_key_operator: str = Field(
        default="operator-secret-key-12345", description="Operator API key"
    )
    api_key_readonly: str = Field(
        default="readonly-secret-key-12345", description="Readonly API key"
    )
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")
    rate_limit_per_minute: int = Field(default=120, description="Rate limit per minute")
    max_query_length: int = Field(default=2000, description="Maximum query length in characters")
    max_document_size_mb: int = Field(default=25, description="Maximum document size in megabytes")

    # Embedding Service
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model name or local path",
    )
    embedding_dimension: int = Field(
        default=512, description="Vector embedding dimension (required 512)"
    )
    embedding_batch_size: int = Field(default=64, description="Embedding inference batch size")
    embedding_normalize: bool = Field(default=True, description="L2 normalize vector embeddings")
    embedding_device: str = Field(default="cpu", description="Compute device: cpu or cuda")

    # FAISS Index
    faiss_index_type: str = Field(default="HNSW", description="Index type: HNSW, IVF, FlatIP")
    faiss_metric: str = Field(default="cosine", description="FAISS metric: cosine, l2, ip")
    faiss_m: int = Field(default=32, description="HNSW M parameter")
    faiss_ef_construction: int = Field(default=200, description="HNSW efConstruction")
    faiss_ef_search: int = Field(default=64, description="HNSW efSearch")
    faiss_nlist: int = Field(default=1024, description="IVF nlist clusters")
    faiss_nprobe: int = Field(default=32, description="IVF nprobe search clusters")

    # Storage and Persistence
    data_dir: Path = Field(default=Path("./data"), description="Base data directory")
    index_dir: Path = Field(default=Path("./data/indexes"), description="Index storage directory")
    index_current_path: Path = Field(
        default=Path("./data/indexes/current"), description="Symlink or path to active index"
    )

    # Retrieval Configuration
    retrieval_semantic_weight: float = Field(
        default=0.7, description="Semantic weight in weighted score fusion"
    )
    retrieval_lexical_weight: float = Field(
        default=0.3, description="Lexical weight in weighted score fusion"
    )
    retrieval_candidate_k: int = Field(
        default=50, description="Candidate pool size retrieved from dense/lexical layers"
    )
    retrieval_final_k: int = Field(default=10, description="Final top-k results returned")
    retrieval_rrf_k: int = Field(default=60, description="RRF constant k")

    # Dynamic Chunking
    chunking_strategy: str = Field(
        default="structure", description="Strategy: token, sentence, semantic, structure"
    )
    chunking_target_tokens: int = Field(default=350, description="Target token size per chunk")
    chunking_min_tokens: int = Field(default=100, description="Minimum token size per chunk")
    chunking_max_tokens: int = Field(default=600, description="Maximum token size per chunk")
    chunking_overlap_tokens: int = Field(default=50, description="Overlap token count")

    # RAG Configuration
    rag_max_context_tokens: int = Field(
        default=2048, description="Maximum total tokens in assembled RAG context"
    )
    rag_max_chunks: int = Field(default=5, description="Maximum chunks included in context")
    rag_max_chunks_per_doc: int = Field(
        default=2, description="Maximum chunks from the same document"
    )
    rag_llm_provider: str = Field(default="local", description="LLM provider: local, openai, mock")

    # Observability
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    prometheus_metrics_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics endpoint"
    )

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Embedding dimension must be positive")
        return v

    @classmethod
    def load(cls, config_file: Path | None = None) -> Settings:
        """Create settings instance with optional YAML override."""
        base = cls()
        if config_file and config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
                # Flatten nested YAML structure if present
                flat: dict[str, Any] = {}
                for section, values in yaml_data.items():
                    if isinstance(values, dict):
                        for k, val in values.items():
                            flat[f"{section}_{k}"] = val
                    else:
                        flat[section] = values
                # Merge into settings
                merged = base.model_dump()
                merged.update({k: v for k, v in flat.items() if k in merged})
                return cls(**merged)
        return base


# Global singleton instance
settings = Settings.load(
    Path("configs/default.yaml") if Path("configs/default.yaml").exists() else None
)
