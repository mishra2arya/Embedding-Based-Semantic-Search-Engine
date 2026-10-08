"""Step-level request latency tracing and context tracking."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class RequestProfiler:
    """Measures fine-grained phase latencies for a search request."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.durations: dict[str, float] = {}

    @contextmanager
    def time_step(self, step_name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.durations[step_name] = (time.perf_counter() - t0) * 1000.0

    def total_elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000.0

    def get_summary(self) -> dict[str, float]:
        summary = {k: round(v, 2) for k, v in self.durations.items()}
        summary["total_ms"] = round(self.total_elapsed_ms(), 2)
        return summary
