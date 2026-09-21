"""Database session management. Defaults to a local SQLite file so the whole platform runs with
zero external services; set DATABASE_URL to point at Postgres in docker-compose/production."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from neuroforge.db.models import Base

logger = logging.getLogger("neuroforge.db")


def get_database_url() -> str:
    return os.environ.get("NEUROFORGE_DATABASE_URL", "sqlite:///./neuroforge.db")


def make_engine(url: str | None = None):
    url = url or get_database_url()
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(url, connect_args=connect_args, future=True)
    if is_sqlite:
        # SQLite ignores foreign keys unless asked per connection. Local dev and CI run on SQLite
        # while production runs Postgres, so without this a bad genome hash in a foreign-key column
        # passes every local test and only fails in production.
        @event.listens_for(engine, "connect")
        def _enforce_foreign_keys(dbapi_connection, _record) -> None:  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


_engine = make_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(engine=None) -> None:
    engine = engine or _engine
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine) -> None:
    """`create_all` creates missing *tables* but never alters existing ones, so a local SQLite
    database created by an earlier version lacked every column added since — and the first query
    that selected one failed with "no such column". Add missing nullable columns in place; a
    missing NOT NULL column can't be added safely, so that case fails loudly with the fix."""
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            if not column.nullable:
                raise RuntimeError(
                    f"table '{table.name}' is missing NOT NULL column '{column.name}'; this database "
                    "predates the current schema — run `alembic upgrade head` (or start from a new "
                    "database)"
                )
            column_type = column.type.compile(engine.dialect)
            with engine.begin() as connection:
                connection.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}')
                )
            logger.warning("added missing column %s.%s to an existing database", table.name, column.name)


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
