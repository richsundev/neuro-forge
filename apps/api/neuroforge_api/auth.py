"""API authentication, authorization, and rate limiting (section 47).

Deliberately simple: API keys are random tokens, stored hashed (passlib pbkdf2_sha256) with a role.
Rate limiting is an in-memory token bucket per key — fine for a single-process demo/portfolio
deployment; a real multi-instance deployment would move this to Redis (see docs/security.md).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
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
        # Endpoints are sync `def`s, so FastAPI runs them on a thread pool: unsynchronized
        # read-modify-write on a bucket could let concurrent requests spend the same token.
        self._lock = threading.Lock()

    MAX_TRACKED_KEYS = 10_000

    def _refilled(self, key: str) -> _Bucket:
        now = time.monotonic()
        if key not in self._buckets and len(self._buckets) >= self.MAX_TRACKED_KEYS:
            # One bucket per client address is otherwise unbounded (an attacker can vary the address
            # cheaply). Buckets that have fully refilled carry no state, so dropping them is lossless.
            full_after = self.capacity / self.refill_per_second
            self._buckets = {k: b for k, b in self._buckets.items() if now - b.last_refill < full_after}
            if len(self._buckets) >= self.MAX_TRACKED_KEYS:
                self._buckets.pop(next(iter(self._buckets)))
        bucket = self._buckets.setdefault(key, _Bucket(tokens=self.capacity))
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.last_refill = now
        return bucket

    def allow(self, key: str) -> bool:
        with self._lock:
            bucket = self._refilled(key)
            if bucket.tokens < 1:
                return False
            bucket.tokens -= 1
            return True

    def blocked(self, key: str) -> bool:
        """True if `allow` would refuse right now (without consuming a token)."""
        with self._lock:
            return self._refilled(key).tokens < 1


_rate_limiter = RateLimiter()
# Failed authentications are throttled per client address. Each failed attempt costs one PBKDF2
# verification per stored key, and the per-key limiter above only ever applied to *valid* keys, so
# nothing slowed down guessing.
_failure_limiter = RateLimiter(capacity=10, refill_per_second=0.2)


# Recently verified keys, by SHA-256 of the raw key. PBKDF2 is deliberately slow and every request
# used to run it against every stored key; a hit here skips that, and — because it is checked
# before the failed-attempt throttle — a valid key keeps working while someone sharing its client
# address (a proxy) is being throttled for guessing.
_VERIFIED_TTL_SECONDS = 300.0
_VERIFIED_MAX_ENTRIES = 1024
_verified: dict[str, tuple[str, str, str, float]] = {}
_verified_lock = threading.Lock()


def _fingerprint(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _cached_principal(fingerprint: str) -> tuple[Principal, str] | None:
    with _verified_lock:
        entry = _verified.get(fingerprint)
        if entry is None:
            return None
        name, role, key_hash, expires_at = entry
        if expires_at < time.monotonic():
            del _verified[fingerprint]
            return None
        return Principal(name=name, role=role), key_hash


def _remember(fingerprint: str, record: ApiKeyRecord) -> None:
    with _verified_lock:
        if len(_verified) >= _VERIFIED_MAX_ENTRIES:
            _verified.pop(next(iter(_verified)))
        _verified[fingerprint] = (
            record.name,
            record.role,
            record.key_hash,
            time.monotonic() + _VERIFIED_TTL_SECONDS,
        )


def get_principal(
    request: Request, raw_key: str | None = Depends(_api_key_header)
) -> Principal:
    client = request.client.host if request.client else "unknown"
    if raw_key:
        fingerprint = _fingerprint(raw_key)
        cached = _cached_principal(fingerprint)
        if cached is not None:
            principal, key_hash = cached
            if not _rate_limiter.allow(key_hash):
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
            return principal
    if _failure_limiter.blocked(client):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many failed authentication attempts")
    if not raw_key:
        _failure_limiter.allow(client)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key header")
    with session_scope() as session:
        candidates = list(session.scalars(select(ApiKeyRecord)))
    for record in candidates:
        if _pwd_context.verify(raw_key, record.key_hash):
            _remember(fingerprint, record)
            if not _rate_limiter.allow(record.key_hash):
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
            return Principal(name=record.name, role=record.role)
    _failure_limiter.allow(client)
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
