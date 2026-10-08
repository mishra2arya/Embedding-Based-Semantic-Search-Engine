"""Security controls: Authentication, Authorization (RBAC), Rate Limiting, and Input Validation."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from enum import StrEnum

from fastapi import Header

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    InvalidQueryError,
    PromptInjectionError,
    RateLimitExceededError,
    TenantIsolationError,
)


class Role(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    READONLY = "readonly"


class RateLimiter:
    """In-memory sliding window rate limiter."""

    def __init__(self, limit_per_minute: int = 120):
        self.limit_per_minute = limit_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - 60.0

        # Prune older timestamps
        self.requests[key] = [t for t in self.requests[key] if t > window_start]

        if len(self.requests[key]) >= self.limit_per_minute:
            raise RateLimitExceededError(
                f"Rate limit of {self.limit_per_minute} requests/minute exceeded. Try again later."
            )

        self.requests[key].append(now)


rate_limiter = RateLimiter(limit_per_minute=settings.rate_limit_per_minute)


def authenticate_api_key(api_key: str | None = Header(None, alias="X-API-Key")) -> tuple[str, Role]:
    """Authenticate incoming request and return (key, Role)."""
    if not api_key:
        raise AuthenticationError("Missing X-API-Key header.")

    if api_key == settings.api_key_admin:
        return api_key, Role.ADMIN
    elif api_key == settings.api_key_operator:
        return api_key, Role.OPERATOR
    elif api_key == settings.api_key_readonly:
        return api_key, Role.READONLY

    raise AuthenticationError("Invalid API key provided.")


def require_role(user_role: Role, allowed_roles: list[Role]) -> None:
    """Verify if user has sufficient privileges."""
    if user_role not in allowed_roles:
        raise AuthorizationError(
            f"Insufficient permissions. Required one of: {[r.value for r in allowed_roles]}"
        )


def validate_query_string(query: str, max_length: int = 2000) -> str:
    """Validate and sanitize user query string."""
    if not query or not query.strip():
        raise InvalidQueryError("Search query cannot be empty or whitespace only.")

    stripped = query.strip()
    if len(stripped) > max_length:
        raise InvalidQueryError(
            f"Query length ({len(stripped)}) exceeds maximum allowed ({max_length})."
        )

    return stripped


# Known adversarial prompt injection attack signatures
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(new\s+)?(assistant|system|persona)", re.IGNORECASE),
    re.compile(r"system\s*:\s*(you\s+(are|must)|override)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|security)\s+filters", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|endoftext\|>", re.IGNORECASE),
]


def detect_prompt_injection(text: str, strict: bool = False) -> bool:
    """Detect common prompt injection attacks in queries or untrusted context."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            if strict:
                raise PromptInjectionError("Potential prompt injection pattern detected in input.")
            return True
    return False


def sanitize_filename(filename: str) -> str:
    """Defend against path traversal attacks in file uploads/loaders."""
    cleaned = os.path.basename(filename)
    # Remove null bytes and non-printable characters
    cleaned = re.sub(r"[^\w\.\-\s]", "_", cleaned)
    if not cleaned or cleaned.startswith("."):
        cleaned = f"doc_{int(time.time())}_{cleaned}"
    return cleaned


def enforce_tenant_isolation(doc_tenant: str | None, requested_tenant: str | None) -> None:
    """Strictly prevent cross-tenant information leakage."""
    if requested_tenant and doc_tenant and doc_tenant != requested_tenant:
        raise TenantIsolationError(
            f"Access denied: document tenant '{doc_tenant}' does not match '{requested_tenant}'."
        )
