import pytest

from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.checkpoint import ExperimentCheckpoint
from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
from neuroforge.experiments.spaces import forge_support_search_space


def _config(experiment_id: str, max_candidates: int, seed: int = 5) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        domain_name="forge-support",
        search_strategy="evolutionary",
        search_space=forge_support_search_space(),
        batch_size=8,
        max_batches=100,
        seed=seed,
        budget=ExperimentBudget(
            max_candidates=max_candidates, max_requests=100_000, max_cost_usd=1000, max_duration_minutes=30
        ),
    )


def test_budget_stops_the_experiment(baseline_genome, small_dataset, tmp_path):
    cfg = _config("exp-budget", max_candidates=8)
    engine = ExperimentEngine(cfg, baseline_genome, small_dataset, tmp_path)
    result = engine.run()
    assert result.candidates_evaluated == 8
    assert "max_candidates" in result.stop_reason


def test_resume_continues_from_checkpoint(baseline_genome, small_dataset, tmp_path):
    state_dir = tmp_path / "resumable"
    cfg_small = _config("exp-resume", max_candidates=16)
    ExperimentEngine(cfg_small, baseline_genome, small_dataset, state_dir).run()

    ckpt_after_first = ExperimentCheckpoint.load(state_dir / "exp-resume.checkpoint.json")
    assert ckpt_after_first.candidates_completed() == 16

    cfg_bigger = _config("exp-resume", max_candidates=32)
    result = ExperimentEngine(cfg_bigger, baseline_genome, small_dataset, state_dir).run()
    assert result.candidates_evaluated == 32


def test_resume_reproduces_uninterrupted_run(baseline_genome, small_dataset, tmp_path):
    """Stopping early and resuming to N candidates must match a single run straight to N,
    because the search strategy's state is fully determined by the sequence of tell() batches."""
    resumed_dir = tmp_path / "resumed"
    ExperimentEngine(_config("exp-a", max_candidates=8), baseline_genome, small_dataset, resumed_dir).run()
    resumed_result = ExperimentEngine(
        _config("exp-a", max_candidates=24), baseline_genome, small_dataset, resumed_dir
    ).run()

    straight_dir = tmp_path / "straight"
    straight_result = ExperimentEngine(
        _config("exp-a", max_candidates=24), baseline_genome, small_dataset, straight_dir
    ).run()

    assert resumed_result.fitness_history == straight_result.fitness_history
    assert resumed_result.best_genome["model"] == straight_result.best_genome["model"]


def test_evaluation_timeout_marks_candidate_failed_but_experiment_continues(
    baseline_genome, small_dataset, tmp_path, monkeypatch
):
    """A hanging domain.evaluate() call must not hang the whole experiment — it should be scored
    as a hard failure for that candidate and the run should continue to completion."""
    import time

    from neuroforge.domains.forge_support import ForgeSupportDomain

    original_evaluate = ForgeSupportDomain.evaluate
    call_count = {"n": 0}

    def slow_evaluate(self, genome, challenge, provider):
        call_count["n"] += 1
        if call_count["n"] == 1:
            time.sleep(2)
        return original_evaluate(self, genome, challenge, provider)

    monkeypatch.setattr(ForgeSupportDomain, "evaluate", slow_evaluate)

    cfg = _config("exp-timeout", max_candidates=8)
    cfg = cfg.model_copy(update={"evaluation_timeout_seconds": 0.2})
    engine = ExperimentEngine(cfg, baseline_genome, small_dataset, tmp_path)
    result = engine.run()
    assert result.candidates_evaluated == 8
    assert 0.0 in result.fitness_history


def test_search_space_outside_mutation_policy_is_rejected(baseline_genome, small_dataset, tmp_path):
    """A search space that reaches outside the mutation policy's allow-list must be refused
    loudly at run time, not silently produce an out-of-policy candidate genome."""
    from neuroforge.mutations.policy import MutationPolicy
    from neuroforge.optimization.search_space import ParamSpec, SearchSpace

    unsafe_space = SearchSpace(parameters={"system_id": ParamSpec(type="categorical", values=["x", "y"])})
    cfg = _config("exp-invalid-candidate", max_candidates=8)
    cfg = cfg.model_copy(update={"search_space": unsafe_space, "mutation_policy": MutationPolicy()})
    engine = ExperimentEngine(cfg, baseline_genome, small_dataset, tmp_path)
    with pytest.raises(RuntimeError, match="outside the mutation policy allow-list"):
        engine.run()


def test_cancel_flag_stops_the_run(baseline_genome, small_dataset, tmp_path):
    cfg = _config("exp-cancel", max_candidates=1000)
    (tmp_path / "exp-cancel.cancel").touch()
    engine = ExperimentEngine(cfg, baseline_genome, small_dataset, tmp_path)
    result = engine.run()
    assert result.stop_reason == "cancelled by user"
    assert result.candidates_evaluated == 0
