"""API authentication, authorization, and rate limiting (section 47).

Deliberately simple: API keys are random tokens, stored hashed (passlib/bcrypt) with a role.
Rate limiting is an in-memory token bucket per key — fine for a single-process demo/portfolio
deployment; a real multi-instance deployment would move this to Redis (see docs/security.md).
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader
from passlib.context import CryptContext
from sqlalchemy import select

from neuroforge.db.models import ApiKeyRecord, AuditLogRecord
from neuroforge.db.session import session_scope

_pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

ROLE_RANK = {"viewer": 0, "operator": 1, "admin": 2}


@dataclass
class Principal:
    name: str
    role: str


def hash_key(raw_key: str) -> str:
    return _pwd_context.hash(raw_key)


def generate_api_key() -> str:
    return f"nf_{secrets.token_urlsafe(32)}"


def bootstrap_default_key() -> str | None:
    """On a totally fresh DB, mint one admin key so local/dev users aren't locked out, and print
    it once. In production, keys should be provisioned out-of-band instead."""
    with session_scope() as session:
        existing = session.scalar(select(ApiKeyRecord).limit(1))
        if existing is not None:
            return None
        raw_key = os.environ.get("NEUROFORGE_BOOTSTRAP_API_KEY") or generate_api_key()
        session.add(ApiKeyRecord(name="bootstrap-admin", key_hash=hash_key(raw_key), role="admin"))
    return raw_key


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    def __init__(self, capacity: int = 60, refill_per_second: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(key, _Bucket(tokens=self.capacity))
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.last_refill = now
        if bucket.tokens < 1:
            return False
        bucket.tokens -= 1
        return True


_rate_limiter = RateLimiter()


def get_principal(
    request: Request, raw_key: str | None = Depends(_api_key_header)
) -> Principal:
    if not raw_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key header")
    with session_scope() as session:
        candidates = list(session.scalars(select(ApiKeyRecord)))
    for record in candidates:
        if _pwd_context.verify(raw_key, record.key_hash):
            if not _rate_limiter.allow(record.key_hash):
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
            return Principal(name=record.name, role=record.role)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")


def require_role(minimum_role: str):
    def _check(principal: Principal = Depends(get_principal)) -> Principal:
        if ROLE_RANK.get(principal.role, -1) < ROLE_RANK[minimum_role]:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role '{principal.role}' cannot perform an action requiring '{minimum_role}'",
            )
        return principal

    return _check


def audit(actor: str, action: str, detail: dict | None = None) -> None:
    with session_scope() as session:
        session.add(AuditLogRecord(actor=actor, action=action, detail=detail or {}))
