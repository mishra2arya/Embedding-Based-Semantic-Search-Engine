"""Adaptive batch processor for vector inference."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def chunk_into_batches(items: list[T], batch_size: int = 64) -> Iterator[list[T]]:
    """Yield successive batch_size-sized chunks from items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
