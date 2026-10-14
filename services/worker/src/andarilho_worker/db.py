"""Plain psycopg 3 access to the API-owned schema (the worker never imports the API)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row


def libpq_url(url: str) -> str:
    """Accept SQLAlchemy-style URLs (postgresql+psycopg://) as well as libpq URLs."""
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


class Database:
    def __init__(self, url: str) -> None:
        self.url = libpq_url(url)

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.url, row_factory=dict_row, autocommit=False, connect_timeout=10)

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection]:
        """One short transaction on a fresh connection; commits on success."""
        with self.connect() as conn:
            yield conn
            conn.commit()

    def ping(self) -> bool:
        with self.connect() as conn:
            conn.execute("SELECT 1")
        return True
