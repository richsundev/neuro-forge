"""Dataset versioning. A dataset version is an immutable, hashable snapshot of challenges split
into train/validation/holdout — see holdout.py for why the holdout split is load-bearing."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from neuroforge.domains.base import Challenge


class DatasetVersion(BaseModel):
    dataset_id: str
    version: int
    parent_version: int | None = None
    source: str  # "seed" | "failure_driven" | "benchmark_evolution"
    challenges: list[Challenge]
    difficulty_score: float
    validated: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def hash(self) -> str:
        # Identity includes the dataset id and version, not just the challenges. The hash is
        # unique in the database, and two datasets generated with the same parameters have identical
        # challenges: with a content-only hash, seeding "beta" after "alpha" (same n and seed — e.g.
        # two applications auto-seeding with the defaults) returned 200 and silently created nothing.
        payload = json.dumps(
            {
                "dataset_id": self.dataset_id,
                "version": self.version,
                "challenges": [c.model_dump(mode="json") for c in self.challenges],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def domain_name(self) -> str:
        """Which domain generated this dataset, from its challenge ids (`fs-`, `sql-`, `ra-`). Kept in
        one place: the API and CLI each had their own copy of this mapping, and evolving a dataset
        trusted a caller-supplied domain instead — so evolving a research-agent dataset with the
        default domain silently mixed ForgeSupport challenges into it."""
        if not self.challenges:
            raise ValueError(f"dataset '{self.dataset_id}' has no challenges")
        prefix = self.challenges[0].challenge_id.split("-")[0]
        return {"fs": "forge-support", "sql": "sql-agent", "ra": "research-agent"}.get(prefix, "forge-support")

    def split(self, name: str) -> list[Challenge]:
        return [c for c in self.challenges if c.split == name]

    def summary(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "hash": self.hash(),
            "source": self.source,
            "n_challenges": len(self.challenges),
            "difficulty_score": self.difficulty_score,
            "splits": {
                name: len(self.split(name)) for name in ("train", "validation", "holdout")
            },
            "validated": self.validated,
        }
