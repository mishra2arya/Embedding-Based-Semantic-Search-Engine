"""FastAPI dependency injection and application state container."""

from __future__ import annotations

import logging

from fastapi import Depends

from app.core.config import settings
from app.core.exceptions import AuthorizationError
from app.core.security import Role, authenticate_api_key
from app.embeddings.model import EmbeddingService
from app.indexing.index_manager import IndexManager
from app.ingestion.pipeline import IngestionPipeline
from app.observability.health import HealthChecker
from app.rag.generator import RAGPipeline
from app.retrieval.hybrid import HybridSearchEngine

logger = logging.getLogger(__name__)


class ApplicationState:
    """Holds shared production singleton services."""

    def __init__(self):
        self.embedding_service: EmbeddingService | None = None
        self.index_manager: IndexManager | None = None
        self.search_engine: HybridSearchEngine | None = None
        self.rag_pipeline: RAGPipeline | None = None
        self.ingestion_pipeline: IngestionPipeline | None = None
        self.health_checker: HealthChecker | None = None

    def initialize(self) -> None:
        """Initialize all services at application startup."""
        logger.info("Initializing application services...")
        self.embedding_service = EmbeddingService(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            normalize=settings.embedding_normalize,
            device=settings.embedding_device,
        )

        self.index_manager = IndexManager(
            base_dir=settings.index_dir,
            dimension=settings.embedding_dimension,
            index_type=settings.faiss_index_type,
            metric=settings.faiss_metric,
        )

        self.search_engine = HybridSearchEngine(
            embedding_service=self.embedding_service,
            index_manager=self.index_manager,
        )
        self.search_engine.sync_lexical_index()

        self.rag_pipeline = RAGPipeline(search_engine=self.search_engine)
        self.ingestion_pipeline = IngestionPipeline()
        self.health_checker = HealthChecker(self.embedding_service, self.index_manager)
        logger.info("Application services initialized successfully.")


# Global state singleton
app_state = ApplicationState()


def get_current_user(
    api_key_role: tuple[str, Role] = Depends(authenticate_api_key),
) -> tuple[str, Role]:
    return api_key_role


def require_admin(user: tuple[str, Role] = Depends(get_current_user)) -> None:
    _, role = user
    if role != Role.ADMIN:
        raise AuthorizationError("Admin privileges required.")


def require_operator(user: tuple[str, Role] = Depends(get_current_user)) -> None:
    _, role = user
    if role not in (Role.ADMIN, Role.OPERATOR):
        raise AuthorizationError("Operator or Admin privileges required.")


def ensure_initialized() -> None:
    if app_state.search_engine is None:
        app_state.initialize()


def get_search_engine() -> HybridSearchEngine:
    ensure_initialized()
    assert app_state.search_engine is not None
    return app_state.search_engine


def get_rag_pipeline() -> RAGPipeline:
    ensure_initialized()
    assert app_state.rag_pipeline is not None
    return app_state.rag_pipeline


def get_ingestion_pipeline() -> IngestionPipeline:
    ensure_initialized()
    assert app_state.ingestion_pipeline is not None
    return app_state.ingestion_pipeline


def get_index_manager() -> IndexManager:
    ensure_initialized()
    assert app_state.index_manager is not None
    return app_state.index_manager


def get_health_checker() -> HealthChecker:
    ensure_initialized()
    assert app_state.health_checker is not None
    return app_state.health_checker


def get_embedding_service() -> EmbeddingService:
    ensure_initialized()
    assert app_state.embedding_service is not None
    return app_state.embedding_service
