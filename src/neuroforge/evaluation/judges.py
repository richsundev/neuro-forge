"""Evaluator ensemble (section 25). A single evaluator is a single opinion; NeuroForge combines
several and reports disagreement rather than silently taking whichever is most favorable."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.domains.base import EvaluationResult
from neuroforge.util import stable_unit_interval

# Each "judge" applies a small deterministic bias/noise perturbation to the deterministic quality
# score, standing in for the disagreement real LLM judges and evaluators actually show.
JUDGE_PROFILES: dict[str, tuple[float, float]] = {
    "deterministic_evaluator": (0.0, 0.0),
    "reference_evaluator": (0.01, 0.02),
    "llm_judge_a": (-0.02, 0.05),
    "llm_judge_b": (0.03, 0.06),
    "embedding_evaluator": (-0.01, 0.03),
}

AGREEMENT_UNCERTAINTY_THRESHOLD = 0.08  # stdev across judges above this => flagged uncertain


@dataclass
class JudgeEnsembleResult:
    challenge_id: str
    genome_hash: str
    per_judge_scores: dict[str, float]
    mean_score: float
    stdev: float
    uncertain: bool


def evaluate_with_ensemble(result: EvaluationResult) -> JudgeEnsembleResult:
    base = result.metrics.get("quality", 0.0)
    scores: dict[str, float] = {}
    for judge, (bias, noise_scale) in JUDGE_PROFILES.items():
        noise = (
            stable_unit_interval(judge, result.genome_hash, result.challenge_id) - 0.5
        ) * 2 * noise_scale
        scores[judge] = min(1.0, max(0.0, base + bias + noise))

    values = list(scores.values())
    mean_score = sum(values) / len(values)
    variance = sum((v - mean_score) ** 2 for v in values) / len(values)
    stdev = variance**0.5

    return JudgeEnsembleResult(
        challenge_id=result.challenge_id,
        genome_hash=result.genome_hash,
        per_judge_scores={k: round(v, 4) for k, v in scores.items()},
        mean_score=round(mean_score, 4),
        stdev=round(stdev, 4),
        uncertain=stdev > AGREEMENT_UNCERTAINTY_THRESHOLD,
    )
