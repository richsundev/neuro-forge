"""Small shared helpers used across packages."""

from __future__ import annotations

import hashlib
import random


def stable_rng(*parts: object) -> random.Random:
    """A random.Random seeded deterministically from its arguments — same arguments, same
    stream, forever. This is how the whole platform stays reproducible without paid APIs."""
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def stable_unit_interval(*parts: object) -> float:
    """A deterministic float in [0, 1) derived from its arguments."""
    return stable_rng(*parts).random()


def stable_int(*parts: object) -> int:
    """A deterministic 32-bit non-negative int derived from its arguments.

    Use this instead of Python's builtin `hash()` anywhere the result must be reproducible
    across processes/runs: `hash()` on str/tuple is salted per-process by PYTHONHASHSEED
    randomization (a security feature, on by default since Python 3.3), so two runs of the same
    experiment with the same seed would otherwise silently produce different mock-provider
    outputs — exactly the kind of bug `make reproduce` exists to catch.
    """
    return stable_rng(*parts).getrandbits(32)


def get_field_by_path(obj: object, path: str) -> object:
    """Resolve a dotted attribute path like 'retrieval.top_k' against a (possibly nested) object."""
    cursor = obj
    for part in path.split("."):
        cursor = getattr(cursor, part)
    return cursor
