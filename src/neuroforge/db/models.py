"""SQLAlchemy models (section 32). Nested domain objects (genome config, experiment results) are
already validated, versioned Pydantic models with their own hashing — persisting them as JSON
columns keyed by their content hash avoids maintaining a second, parallel relational schema for
the same data. Relational columns are used for everything the API actually queries/filters by."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


class ApplicationRecord(Base):
    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    domain_name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class GenomeRecord(Base):
    __tablename__ = "system_genomes"

    hash: Mapped[str] = mapped_column(String(32), primary_key=True)
    system_id: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int] = mapped_column(Integer)
    parent_hash: Mapped[str | None] = mapped_column(String(32), ForeignKey("system_genomes.hash"), nullable=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="GENERATED")
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_genome_system_version", "system_id", "version"),)


class DatasetVersionRecord(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int] = mapped_column(Integer)
    parent_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50))
    dataset_hash: Mapped[str] = mapped_column(String(32), unique=True)
    difficulty_score: Mapped[float] = mapped_column(Float)
    n_challenges: Mapped[int] = mapped_column(Integer)
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_dataset_id_version", "dataset_id", "version"),)


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    application_id: Mapped[str] = mapped_column(String(64), index=True)
    domain_name: Mapped[str] = mapped_column(String(100))
    search_strategy: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="running")
    config: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class PromotionRecord(Base):
    __tablename__ = "promotion_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    genome_hash: Mapped[str] = mapped_column(String(32), ForeignKey("system_genomes.hash"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    approved: Mapped[bool] = mapped_column(default=False)
    next_status: Mapped[str] = mapped_column(String(30))
    reasons: Mapped[dict] = mapped_column(JSON)
    evidence: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class HoldoutEvaluationRecord(Base):
    """The one-and-only holdout measurement of a candidate against a dataset version. Persisting it
    (rather than re-evaluating on every promotion request) is what makes "the holdout is evaluated
    at most once per candidate" true across processes and repeated requests, not just within one
    HoldoutGuard instance."""

    __tablename__ = "holdout_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    genome_hash: Mapped[str] = mapped_column(String(32), ForeignKey("system_genomes.hash"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dataset_id: Mapped[str] = mapped_column(String(100))
    dataset_version: Mapped[int] = mapped_column(Integer)
    evidence: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("genome_hash", "dataset_id", "dataset_version", name="uq_holdout_once"),
    )


class PromotionApprovalRecord(Base):
    """A human's explicit sign-off that moves a genome to PROMOTED — separate from
    PromotionRecord, which logs the automated gate decision. Requiring both an approved gate
    decision *and* a row here is what makes PROMOTED reachable only through an explicit,
    attributable human action (see docs/promotion.md)."""

    __tablename__ = "promotion_approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    genome_hash: Mapped[str] = mapped_column(String(32), ForeignKey("system_genomes.hash"), index=True)
    experiment_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    approved_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CanaryRecord(Base):
    __tablename__ = "canary_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    baseline_hash: Mapped[str] = mapped_column(String(32))
    candidate_hash: Mapped[str] = mapped_column(String(32), index=True)
    traffic_split: Mapped[float] = mapped_column(Float)
    rollback_triggered: Mapped[bool] = mapped_column(default=False)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ApiKeyRecord(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(200), unique=True)
    role: Mapped[str] = mapped_column(String(30), default="viewer")  # viewer | operator | admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLogRecord(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(100))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
