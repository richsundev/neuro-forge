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
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from neuroforge.genomes.schema import SystemGenome


class TellBatch(BaseModel):
    points: list[dict[str, Any]]
    fitness: list[float]


class ExperimentCheckpoint(BaseModel):
    experiment_id: str
    strategy_name: str
    seed: int
    status: str = "running"  # running | completed | cancelled
    stop_reason: str = ""
    budget_state: dict[str, float] = Field(default_factory=dict)
    tell_batches: list[TellBatch] = Field(default_factory=list)
    best_genome: dict[str, Any] | None = None
    best_fitness: float = float("-inf")
    # Selection is feasible-first (see engine.py), so resuming needs to know whether the stored
    # best already clears the constraints, not just its fitness.
    best_feasible: bool = False
    dataset_version_hash: str = ""
    # The dataset version the search started on. A resumed run must keep using it: batches evaluated
    # on v1 and batches evaluated on an evolved v2 aren't comparable, and the fitness history would
    # silently mix them. 0 = written before this field existed (not enforced).
    dataset_version: int = 0

    @field_validator("best_fitness", mode="before")
    @classmethod
    def _null_means_no_best_yet(cls, value: Any) -> Any:
        # `best_fitness` starts at -inf, which pydantic writes as JSON `null` — and then refuses to
        # read back into a float. An experiment stopped before its first generation (cancelled, or
        # out of budget) saved exactly that and could never be loaded, resumed or inspected again.
        return float("-inf") if value is None else value

    def generations_completed(self) -> int:
        return len(self.tell_batches)

    def candidates_completed(self) -> int:
        return sum(len(b.points) for b in self.tell_batches)

    def save(self, path: Path) -> None:
        # Write-then-rename so a crash mid-write can never leave a truncated checkpoint behind: the
        # reader sees either the previous complete file or the new complete one.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        os.replace(tmp, path)

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
