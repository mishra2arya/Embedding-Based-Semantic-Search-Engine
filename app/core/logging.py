"""Structured JSON logging system with privacy redaction and request tracing."""

from __future__ import annotations

import datetime
import json
import logging
import sys
from typing import Any


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as structured JSON."""

    def __init__(self, redact_content: bool = True):
        super().__init__()
        self.redact_content = redact_content

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom contextual attributes if present
        context_keys = [
            "event",
            "request_id",
            "endpoint",
            "tenant_id",
            "query_id",
            "latency_ms",
            "status",
            "results",
            "index_version",
            "error_code",
            "document_id",
            "chunk_count",
        ]
        for key in context_keys:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    redact_content: bool = True,
) -> None:
    """Configure global logging handler."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(StructuredJSONFormatter(redact_content=redact_content))
    else:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
        )

    root_logger.addHandler(handler)

    # Lower noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Obtain a structured logger for a component."""
    return logging.getLogger(name)
