"""Experiment checkpointing and resumption (sections 20/35).

The checkpoint stores every (point, fitness) observation ever told to the search strategy, grouped
by the batch it was told in. Resuming an experiment means: reconstruct a fresh strategy instance
with the same config/seed, replay each historical `tell()` batch in original order (which
reproduces evolutionary population state, GP training data, bandit arm statistics, etc. exactly),
then keep asking for new batches from wherever that leaves off. This makes every strategy
resumable through one mechanism instead of a bespoke serializer per strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from neuroforge.genomes.schema import SystemGenome


class TellBatch(BaseModel):
    points: list[dict[str, Any]]
    fitness: list[float]


class ExperimentCheckpoint(BaseModel):
    experiment_id: str
    strategy_name: str
    seed: int
    status: str = "running"  # running | completed | stopped
    stop_reason: str = ""
    budget_state: dict[str, float] = Field(default_factory=dict)
    tell_batches: list[TellBatch] = Field(default_factory=list)
    best_genome: dict[str, Any] | None = None
    best_fitness: float = float("-inf")
    dataset_version_hash: str = ""

    def generations_completed(self) -> int:
        return len(self.tell_batches)

    def candidates_completed(self) -> int:
        return sum(len(b.points) for b in self.tell_batches)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> ExperimentCheckpoint:
        return cls.model_validate(json.loads(path.read_text()))

    @classmethod
    def load_or_none(cls, path: Path) -> ExperimentCheckpoint | None:
        if not path.exists():
            return None
        return cls.load(path)

    def best_genome_obj(self) -> SystemGenome | None:
        return SystemGenome.model_validate(self.best_genome) if self.best_genome else None
