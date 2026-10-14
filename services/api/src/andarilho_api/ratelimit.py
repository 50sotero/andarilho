"""A small in-process sliding-window rate limiter for the public delivery endpoint.

It bounds abuse from a single client within one API process. Multiple API
replicas each enforce their own window; put a shared limiter at the edge for a
global guarantee.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        max_keys: int = 10000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Record a hit; return ``None`` if allowed, else seconds until retry."""
        with self._lock:
            now = self._clock()
            cutoff = now - self.window
            if len(self._hits) >= self.max_keys:
                self._purge(cutoff)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return None

    def _purge(self, cutoff: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]
        if len(self._hits) >= self.max_keys:
            # Still full of active clients: drop the oldest half rather than grow.
            for key in list(self._hits)[: self.max_keys // 2]:
                del self._hits[key]
