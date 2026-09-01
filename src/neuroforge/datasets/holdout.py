"""Holdout protection: the single most important anti-overfitting mechanism in NeuroForge.

Section 15/64 of the brief is explicit: a candidate must never be selected because it does well on
data the optimizer also searched against, and the code must not merely describe this — it must be
impossible to accidentally violate. `HoldoutGuard` is the enforcement point: every optimizer and
mutation-scoring path is required to fetch challenges through it, and it raises rather than
silently filtering if anyone asks for the holdout split outside of `evaluate_holdout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neuroforge.domains.base import Challenge
from neuroforge.util import stable_rng


class HoldoutViolation(RuntimeError):
    pass


def deterministic_split(
    challenges: list[Challenge],
    seed: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> list[Challenge]:
    """Assign each challenge to train/validation/holdout via a stable per-challenge hash, so the
    same challenge always lands in the same split even as the dataset grows."""
    train_r, val_r, _holdout_r = ratios
    out = []
    for c in challenges:
        r = stable_rng("split", seed, c.challenge_id).random()
        if r < train_r:
            split = "train"
        elif r < train_r + val_r:
            split = "validation"
        else:
            split = "holdout"
        out.append(c.model_copy(update={"split": split}))
    return out


@dataclass
class HoldoutGuard:
    """Wraps a dataset version's challenges and enforces that `holdout` is only reachable through
    `evaluate_holdout`, never through `search_set` / `validation_set` used during optimization."""

    all_challenges: list[Challenge]
    _holdout_accessed_for: set[str] = field(default_factory=set)

    def search_set(self) -> list[Challenge]:
        """Challenges the mutation/search loop is allowed to score candidates against."""
        return [c for c in self.all_challenges if c.split == "train"]

    def validation_set(self) -> list[Challenge]:
        """Used for early-stopping / convergence checks — still not the holdout."""
        return [c for c in self.all_challenges if c.split == "validation"]

    def evaluate_holdout(self, genome_hash: str) -> list[Challenge]:
        """The only sanctioned way to read holdout challenges — for a *final* promotion check on
        one specific candidate, at most once per candidate."""
        if genome_hash in self._holdout_accessed_for:
            raise HoldoutViolation(
                f"Holdout set already evaluated for genome {genome_hash}. Re-running holdout "
                "evaluation to chase a better number would defeat its purpose."
            )
        self._holdout_accessed_for.add(genome_hash)
        return [c for c in self.all_challenges if c.split == "holdout"]
