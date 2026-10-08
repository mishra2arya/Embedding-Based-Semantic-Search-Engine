"""Custom exceptions and standardized error definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    request_id: str | None = Field(default=None, description="Unique request tracing ID")
    details: dict[str, Any] | None = Field(default=None, description="Optional diagnostic details")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class AppException(Exception):
    """Base application exception."""

    status_code: int = 500
    code: str = "INTERNAL_SERVER_ERROR"

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}


class AuthenticationError(AppException):
    status_code = 401
    code = "AUTHENTICATION_REQUIRED"


class AuthorizationError(AppException):
    status_code = 403
    code = "FORBIDDEN"


class RateLimitExceededError(AppException):
    status_code = 429
    code = "RATE_LIMIT_EXCEEDED"


class InvalidQueryError(AppException):
    status_code = 400
    code = "INVALID_QUERY"


class PayloadTooLargeError(AppException):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


class DocumentParsingError(AppException):
    status_code = 422
    code = "DOCUMENT_PARSING_ERROR"


class DocumentNotFoundError(AppException):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"


class IndexNotFoundError(AppException):
    status_code = 404
    code = "INDEX_NOT_FOUND"


class IndexBuildError(AppException):
    status_code = 500
    code = "INDEX_BUILD_ERROR"


class DimensionMismatchError(AppException):
    status_code = 500
    code = "DIMENSION_MISMATCH"


class TenantIsolationError(AppException):
    status_code = 403
    code = "TENANT_ACCESS_DENIED"


class PromptInjectionError(AppException):
    status_code = 400
    code = "PROMPT_INJECTION_DETECTED"


class ServiceUnavailableError(AppException):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"
