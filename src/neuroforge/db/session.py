"""Database session management. Defaults to a local SQLite file so the whole platform runs with
zero external services; set DATABASE_URL to point at Postgres in docker-compose/production."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from neuroforge.db.models import Base


def get_database_url() -> str:
    return os.environ.get("NEUROFORGE_DATABASE_URL", "sqlite:///./neuroforge.db")


def make_engine(url: str | None = None):
    url = url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


_engine = make_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(engine=None) -> None:
    Base.metadata.create_all(engine or _engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
