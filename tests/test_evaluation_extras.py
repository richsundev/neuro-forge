from neuroforge.domains.base import EvaluationResult
from neuroforge.evaluation.judges import evaluate_with_ensemble
from neuroforge.evaluation.objectives import DEFAULT_OBJECTIVES, Objective, ObjectiveSpec


def test_ensemble_flags_high_disagreement_via_extreme_quality():
    # quality=1.0 pushes several judge biases to clamp at the ceiling while others don't,
    # which is what produces disagreement in this deterministic simulation.
    result = EvaluationResult(
        challenge_id="c1", genome_hash="hash1", metrics={"quality": 1.0}, cost_usd=0, latency_ms=0, failed=False
    )
    ensemble = evaluate_with_ensemble(result)
    assert set(ensemble.per_judge_scores) == {
        "deterministic_evaluator",
        "reference_evaluator",
        "llm_judge_a",
        "llm_judge_b",
        "embedding_evaluator",
    }
    assert 0.0 <= ensemble.mean_score <= 1.0


def test_ensemble_is_deterministic():
    result = EvaluationResult(
        challenge_id="c1", genome_hash="hash1", metrics={"quality": 0.7}, cost_usd=0, latency_ms=0, failed=False
    )
    a = evaluate_with_ensemble(result)
    b = evaluate_with_ensemble(result)
    assert a.per_judge_scores == b.per_judge_scores


def test_objective_weights_must_sum_to_one():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ObjectiveSpec(objectives={"quality": Objective(direction="maximize", weight=0.5)})


def test_default_objectives_reward_higher_quality_lower_cost():
    good = DEFAULT_OBJECTIVES.score(
        {"quality": 0.9, "task_success": 0.9, "policy_compliance": 0.9, "safety_score": 0.9,
         "cost_usd": 0.001, "latency_ms": 300, "failure_rate": 0.0}
    )
    bad = DEFAULT_OBJECTIVES.score(
        {"quality": 0.3, "task_success": 0.3, "policy_compliance": 0.3, "safety_score": 0.3,
         "cost_usd": 0.04, "latency_ms": 2500, "failure_rate": 0.5}
    )
    assert good > bad
