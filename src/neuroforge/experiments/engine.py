"""ExperimentEngine: the orchestrator that runs the core loop end to end (section 34).

Search space parameters are genome field paths (e.g. "retrieval.top_k"), so every candidate a
strategy proposes becomes a real, hashed, lineaged `SystemGenome` child of the baseline via
`SystemGenome.derive()` — random search, evolutionary search, and Bayesian optimization all
produce first-class genomes with mutation records, not opaque parameter dicts. Search happens only
against the dataset's train split (`HoldoutGuard.search_set`); final candidate selection also
checks the validation split; the holdout split is reserved for the promotion pipeline
(`promotion/holdout.py`), which evaluates it once per candidate.

Search is constraint-aware: a candidate's fitness is shrunk in proportion to how far it sits
outside the safety limits and the quality floor (each tightened by a small search margin) and the
promotion gates' cost/latency limits, and selection is feasible-first — a feasible candidate always beats
an infeasible one regardless of raw fitness. Without this the optimizer happily converges on the
boundary of (or outside) the limits, and the winner passes or fails the later checks on sampling
noise alone.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

from pydantic import BaseModel, Field

from neuroforge.datasets.dataset import DatasetVersion
from neuroforge.datasets.holdout import HoldoutGuard
from neuroforge.domains import get_domain
from neuroforge.domains.base import Challenge, EvaluationResult
from neuroforge.evaluation.aggregate import AggregateMetrics, aggregate
from neuroforge.evaluation.objectives import DEFAULT_OBJECTIVES, ObjectiveSpec
from neuroforge.evaluation.statistics import ComparisonResult, compare
from neuroforge.experiments.budget import BudgetTracker, ExperimentBudget
from neuroforge.experiments.checkpoint import ExperimentCheckpoint, TellBatch
from neuroforge.experiments.events import Event, EventLog
from neuroforge.genomes.schema import MutationRecord, SystemGenome
from neuroforge.mutations.policy import MutationPolicy
from neuroforge.observability import get_tracer
from neuroforge.optimization import Observation, SearchSpace, get_strategy
from neuroforge.promotion.gates import PromotionGateConfig, metric_gate_findings
from neuroforge.promotion.holdout import per_challenge_metrics
from neuroforge.promotion.safety import SafetyConstraints, SafetyEstimate, estimate_safety
from neuroforge.providers.registry import get_provider
from neuroforge.util import get_field_by_path


class ExperimentConfig(BaseModel):
    experiment_id: str
    domain_name: str
    provider_name: str = "mock"
    search_strategy: str = "evolutionary"
    search_space: SearchSpace
    objectives: ObjectiveSpec = DEFAULT_OBJECTIVES
    budget: ExperimentBudget = Field(default_factory=ExperimentBudget)
    batch_size: int = 8
    max_batches: int = 15
    seed: int = 42
    safety_constraints: SafetyConstraints = Field(default_factory=SafetyConstraints)
    promotion_gates: PromotionGateConfig = Field(default_factory=PromotionGateConfig)
    mutation_policy: MutationPolicy = Field(default_factory=MutationPolicy)
    min_relative_improvement: float = 0.02
    confidence: float = 0.95
    # If set, bounds each candidate's full evaluation (all challenges). A candidate that times out
    # is scored as a hard failure (fitness-worst) and the experiment continues — see
    # docs/development.md's failure-injection section and `make failure-evaluator-timeout`.
    evaluation_timeout_seconds: float | None = None
    # Which dataset the experiment searches/validates against; None means the legacy convention
    # `<application_id>-dataset`. Recorded so a later promotion review evaluates the holdout of
    # the *same* dataset version this experiment used.
    dataset_id: str | None = None
    # Search aims this far inside the *safety* limits so the winner has headroom against
    # split-to-split sampling noise (cost/latency/quality limits are near-deterministic functions
    # of the configuration, so they need no such buffer).
    search_safety_margin: float = Field(default=0.03, ge=0.0, le=0.2)
    # Same idea for the quality floor: the optimum of a cost-weighted objective sits on the floor
    # (cheaper is better right up to the limit), so without headroom the winner lands on it and
    # passes or fails the holdout on noise.
    search_quality_margin: float = Field(default=0.02, ge=0.0, le=0.2)
    # Fitness is divided by (1 + constraint_penalty * violation): 0 disables shaping.
    constraint_penalty: float = Field(default=10.0, ge=0.0)


class ExperimentResult(BaseModel):
    experiment_id: str
    status: str
    stop_reason: str
    generations_completed: int
    candidates_evaluated: int
    baseline_genome: dict
    best_genome: dict
    baseline_metrics: dict[str, float]
    best_metrics: dict[str, float]
    comparison: dict
    budget_utilization: dict[str, float]
    fitness_history: list[float]
    recommendation: str
    dataset_id: str = ""
    dataset_version: int = 0
    # Whether the selected genome cleared the margin-tightened constraints on the search split.
    selected_feasible: bool = False
    validation_safety: dict = Field(default_factory=dict)


class ExperimentEngine:
    def __init__(
        self,
        config: ExperimentConfig,
        baseline_genome: SystemGenome,
        dataset_version: DatasetVersion,
        state_dir: Path,
    ) -> None:
        self.config = config
        self.baseline_genome = baseline_genome
        self.dataset_version = dataset_version
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.state_dir / f"{config.experiment_id}.checkpoint.json"
        self.event_log = EventLog(self.state_dir / f"{config.experiment_id}.events.jsonl")
        self.provider = get_provider(config.provider_name)
        self.domain = get_domain(config.domain_name)
        self.holdout_guard = HoldoutGuard(dataset_version.challenges)
        self._tracer = get_tracer("neuroforge.experiments")

    def run(self) -> ExperimentResult:
        with self._tracer.start_as_current_span(
            "experiment.run", attributes={"experiment_id": self.config.experiment_id, "strategy": self.config.search_strategy}
        ):
            return self._run()

    def _run(self) -> ExperimentResult:
        ckpt = ExperimentCheckpoint.load_or_none(self.checkpoint_path)
        resuming = ckpt is not None
        if not resuming:
            ckpt = ExperimentCheckpoint(
                experiment_id=self.config.experiment_id,
                strategy_name=self.config.search_strategy,
                seed=self.config.seed,
                dataset_version_hash=self.dataset_version.hash(),
                best_genome=self.baseline_genome.model_dump(mode="json"),
                best_fitness=float("-inf"),
            )
            self.event_log.emit(
                Event(
                    "ExperimentCreated",
                    self.config.experiment_id,
                    {
                        "baseline_genome": self.baseline_genome.short_id(),
                        "strategy": self.config.search_strategy,
                        "dataset_version_hash": self.dataset_version.hash(),
                        "seed": self.config.seed,
                    },
                )
            )
        assert ckpt is not None

        strategy = get_strategy(
            self.config.search_strategy,
            self.config.search_space,
            seed=self.config.seed,
            batch_size=self.config.batch_size,
        )
        for batch in ckpt.tell_batches:
            strategy.tell(
                [Observation(point=p, fitness=f) for p, f in zip(batch.points, batch.fitness, strict=True)]
            )

        budget_tracker = BudgetTracker.from_state(self.config.budget, ckpt.budget_state)
        best_genome = ckpt.best_genome_obj() or self.baseline_genome
        best_fitness = ckpt.best_fitness
        best_feasible = ckpt.best_feasible
        next_version = self.baseline_genome.version + 1 + ckpt.candidates_completed()
        generation_index = ckpt.generations_completed()
        search_challenges = self.holdout_guard.search_set()
        search_categories = [c.category for c in search_challenges]
        baseline_search_agg = aggregate(
            [self.domain.evaluate(self.baseline_genome, c, self.provider) for c in search_challenges],
            search_categories,
        )
        search_gates = self.config.promotion_gates.model_copy(
            update={
                "min_quality": min(
                    1.0, self.config.promotion_gates.min_quality + self.config.search_quality_margin
                )
            }
        )
        search_safety = self.config.safety_constraints.tightened(self.config.search_safety_margin)

        stop_reason = ""
        cancel_flag = self.state_dir / f"{self.config.experiment_id}.cancel"
        while True:
            if cancel_flag.exists():
                stop_reason = "cancelled by user"
                cancel_flag.unlink(missing_ok=True)
                break
            exhausted, reason = budget_tracker.exhausted()
            if exhausted:
                stop_reason = reason
                break
            if generation_index >= self.config.max_batches:
                stop_reason = "max_batches reached"
                break

            points = strategy.ask(self.config.batch_size)
            if not points:
                stop_reason = "search space exhausted"
                break

            # Search-space points set every dimension at once (standard HPO), unlike the
            # MutationEngine's incremental deltas — so we only enforce the policy's field
            # allow-list here, not its `max_changes_per_candidate` bound (see docs/design-decisions.md).
            for point in points:
                disallowed = [f for f in point if not self.config.mutation_policy.is_allowed(f)]
                if disallowed:
                    raise RuntimeError(
                        f"search space touches fields outside the mutation policy allow-list: {disallowed}"
                    )

            genomes: list[SystemGenome] = []
            for point in points:
                mutation_records = [
                    MutationRecord(
                        mutation_type=f"{self.config.search_strategy}.propose",
                        field_path=field_path,
                        old_value=get_field_by_path(self.baseline_genome, field_path),
                        new_value=value,
                        reason=(
                            f"Proposed by {self.config.search_strategy} search to optimize the "
                            "configured multi-objective fitness function."
                        ),
                    )
                    for field_path, value in point.items()
                ]
                genome = self.baseline_genome.derive(
                    mutations=mutation_records,
                    overrides=point,
                    new_version=next_version,
                    generation=generation_index,
                )
                next_version += 1
                genomes.append(genome)
                self.event_log.emit(
                    Event(
                        "CandidateGenerated",
                        self.config.experiment_id,
                        {"genome_hash": genome.hash(), "version": genome.version, "point": point},
                    )
                )

            observations: list[Observation] = []
            fitness_values: list[float] = []
            with self._tracer.start_as_current_span(
                "experiment.generation",
                attributes={"generation": generation_index, "population_size": len(genomes)},
            ):
                for genome, point in zip(genomes, points, strict=True):
                    results, timed_out = self._evaluate_candidate(genome, search_challenges)
                    agg = aggregate(results, search_categories)
                    violation = sum(
                        f.magnitude
                        for f in metric_gate_findings(
                            baseline_search_agg, agg, search_gates, search_safety
                        )
                    )
                    feasible = violation == 0.0 and not timed_out
                    base_fitness = self.config.objectives.score(agg.as_dict())
                    fitness = (
                        0.0
                        if timed_out
                        else base_fitness / (1.0 + self.config.constraint_penalty * violation)
                    )
                    observations.append(Observation(point=point, fitness=fitness))
                    fitness_values.append(fitness)
                    budget_tracker.record(
                        candidates=1,
                        requests=len(search_challenges),
                        cost=sum(r.cost_usd for r in results),
                    )
                    self.event_log.emit(
                        Event(
                            "CandidateEvaluated",
                            self.config.experiment_id,
                            {
                                "genome_hash": genome.hash(),
                                "version": genome.version,
                                "generation": generation_index,
                                "fitness": fitness,
                                "metrics": agg.as_dict(),
                                "timed_out": timed_out,
                                "feasible": feasible,
                                "constraint_violation": round(violation, 6),
                            },
                        )
                    )
                    if (feasible, fitness) > (best_feasible, best_fitness):
                        best_fitness = fitness
                        best_feasible = feasible
                        best_genome = genome

            strategy.tell(observations)
            ckpt.tell_batches.append(TellBatch(points=points, fitness=fitness_values))
            generation_index += 1
            self.event_log.emit(
                Event(
                    "GenerationCompleted",
                    self.config.experiment_id,
                    {
                        "generation": generation_index,
                        "mean_fitness": sum(fitness_values) / len(fitness_values),
                        "best_fitness": best_fitness,
                    },
                )
            )

            ckpt.budget_state = budget_tracker.to_state()
            ckpt.best_genome = best_genome.model_dump(mode="json")
            ckpt.best_fitness = best_fitness
            ckpt.best_feasible = best_feasible
            ckpt.save(self.checkpoint_path)

            # A plateau only means "done" once there is a feasible incumbent: while every
            # candidate so far violates the limits, a flat fitness curve is the search still
            # stuck outside the feasible region, not converged on an answer.
            convergence = strategy.convergence()
            if convergence.converged and best_feasible:
                stop_reason = convergence.reason
                break

        ckpt.status = "completed"
        ckpt.stop_reason = stop_reason
        ckpt.save(self.checkpoint_path)
        self.event_log.emit(
            Event("ExperimentStopped", self.config.experiment_id, {"reason": stop_reason})
        )

        return self._finalize(ckpt, best_genome, budget_tracker, best_feasible)

    def _evaluate_candidate(
        self, genome: SystemGenome, challenges: list[Challenge]
    ) -> tuple[list[EvaluationResult], bool]:
        """Evaluate one candidate against every challenge, optionally bounded by
        `evaluation_timeout_seconds`. A timeout is treated as a hard failure for that candidate
        (worst-possible score) rather than crashing the whole experiment — see
        `make failure-evaluator-timeout` and docs/development.md."""
        if self.config.evaluation_timeout_seconds is None:
            return [self.domain.evaluate(genome, c, self.provider) for c in challenges], False

        # Not a `with` block deliberately: ThreadPoolExecutor.__exit__ calls shutdown(wait=True),
        # which would block on the runaway thread and defeat the timeout entirely. shutdown(wait=
        # False) below abandons it instead — it keeps running in the background (Python has no
        # safe way to kill a thread), but this call returns immediately either way.
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            lambda: [self.domain.evaluate(genome, c, self.provider) for c in challenges]
        )
        try:
            result = future.result(timeout=self.config.evaluation_timeout_seconds)
            executor.shutdown(wait=False)
            return result, False
        except FutureTimeoutError:
            executor.shutdown(wait=False)
            worst = [
                EvaluationResult(
                    challenge_id=c.challenge_id,
                    genome_hash=genome.hash(),
                    metrics={"quality": 0.0},
                    cost_usd=0.0,
                    latency_ms=self.config.evaluation_timeout_seconds * 1000,
                    failed=True,
                    notes="evaluator timeout",
                )
                for c in challenges
            ]
            return worst, True

    def _finalize(
        self,
        ckpt: ExperimentCheckpoint,
        best_genome: SystemGenome,
        budget_tracker: BudgetTracker,
        best_feasible: bool,
    ) -> ExperimentResult:
        val_challenges = self.holdout_guard.validation_set()
        baseline_results = [
            self.domain.evaluate(self.baseline_genome, c, self.provider) for c in val_challenges
        ]
        best_results = [self.domain.evaluate(best_genome, c, self.provider) for c in val_challenges]
        categories = [c.category for c in val_challenges]

        baseline_agg = aggregate(baseline_results, categories)
        best_agg = aggregate(best_results, categories)

        baseline_scores = [r.metrics.get("quality", 0.0) for r in baseline_results]
        best_scores = [r.metrics.get("quality", 0.0) for r in best_results]
        comparison = compare(
            baseline_scores,
            best_scores,
            min_relative_improvement=self.config.min_relative_improvement,
            confidence=self.config.confidence,
            seed=self.config.seed,
        )
        safety = estimate_safety(
            per_challenge_metrics(best_results),
            confidence=self.config.confidence,
            seed=self.config.seed,
        )

        fitness_history = [
            f for batch in ckpt.tell_batches for f in batch.fitness
        ]

        recommendation = self._recommend(comparison, baseline_agg, best_agg, safety)

        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            status=ckpt.status,
            stop_reason=ckpt.stop_reason,
            generations_completed=ckpt.generations_completed(),
            candidates_evaluated=ckpt.candidates_completed(),
            baseline_genome=self.baseline_genome.model_dump(mode="json"),
            best_genome=best_genome.model_dump(mode="json"),
            baseline_metrics=baseline_agg.as_dict(),
            best_metrics=best_agg.as_dict(),
            comparison=_comparison_dict(comparison),
            budget_utilization=budget_tracker.utilization(),
            fitness_history=fitness_history,
            recommendation=recommendation,
            dataset_id=self.dataset_version.dataset_id,
            dataset_version=self.dataset_version.version,
            selected_feasible=best_feasible,
            validation_safety=safety.model_dump(mode="json"),
        )

    def _recommend(
        self,
        comparison: ComparisonResult,
        baseline_agg: AggregateMetrics,
        best_agg: AggregateMetrics,
        safety: SafetyEstimate,
    ) -> str:
        from neuroforge.evaluation.statistics import Conclusion

        safety_ok, violations = safety.check(self.config.safety_constraints)
        if not safety_ok:
            return f"DO NOT PROMOTE — safety constraint violated: {'; '.join(violations)}"
        if comparison.conclusion == Conclusion.LIKELY_IMPROVEMENT:
            findings = metric_gate_findings(
                baseline_agg,
                best_agg,
                self.config.promotion_gates,
                self.config.safety_constraints,
                safety,
            )
            if findings:
                return f"DO NOT PROMOTE — promotion gate not met: {'; '.join(f.reason for f in findings)}"
            return "PROMOTE TO CANARY — statistically significant improvement on validation set"
        if comparison.conclusion == Conclusion.LIKELY_REGRESSION:
            return "REJECT — candidate regressed relative to baseline"
        return "INCONCLUSIVE — insufficient evidence of improvement, keep searching or gather more data"


def _comparison_dict(comparison: ComparisonResult) -> dict:
    return {
        "mean_diff": comparison.mean_diff,
        "relative_diff": comparison.relative_diff,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "effect_size": comparison.effect_size,
        "confidence": comparison.confidence,
        "conclusion": comparison.conclusion.value,
        "summary": comparison.summary(),
    }
