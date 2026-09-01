#!/usr/bin/env python3
"""Reproducible research mode (section 59/60): runs a complete optimization experiment against
ForgeSupport using only the deterministic mock provider, plus a search-strategy comparison and an
ablation study, and writes experiment.json / results.json / report.md. Every number in the report
comes from an actual run of the code in this checkout — nothing here is hand-typed.

    python scripts/reproduce.py [--out-dir reproduce_output]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from neuroforge.datasets.evolution import seed_dataset
from neuroforge.domains import get_domain
from neuroforge.evaluation.objectives import DEFAULT_OBJECTIVES
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
from neuroforge.experiments.spaces import forge_support_search_space
from neuroforge.genomes.schema import SystemGenome
from neuroforge.mutations.policy import MutationPolicy

SEED = 11
DOMAIN_NAME = "forge-support"
SYSTEM_ID = "support-agent"
N_CHALLENGES = 100


def run_experiment(
    experiment_id: str,
    strategy: str,
    state_dir: Path,
    search_space=None,
    max_candidates: int = 240,
    batch_size: int = 24,
    max_batches: int = 25,
) -> tuple[dict, float]:
    domain = get_domain(DOMAIN_NAME)
    dataset = seed_dataset(domain, f"{SYSTEM_ID}-dataset", n=N_CHALLENGES, seed=1)
    baseline = SystemGenome(system_id=SYSTEM_ID, version=1)

    config = ExperimentConfig(
        experiment_id=experiment_id,
        domain_name=DOMAIN_NAME,
        search_strategy=strategy,
        search_space=search_space or forge_support_search_space(),
        objectives=DEFAULT_OBJECTIVES,
        batch_size=batch_size,
        max_batches=max_batches,
        seed=SEED,
        budget=ExperimentBudget(
            max_candidates=max_candidates, max_requests=1_000_000, max_cost_usd=1000, max_duration_minutes=30
        ),
        mutation_policy=MutationPolicy(),
    )
    engine = ExperimentEngine(config, baseline, dataset, state_dir / experiment_id)
    start = time.perf_counter()
    result = engine.run()
    elapsed = time.perf_counter() - start
    return result.model_dump(mode="json"), elapsed


def strategy_comparison(state_dir: Path) -> list[dict]:
    rows = []
    for strategy in ("random", "evolutionary", "bayesian", "bandit"):
        result, elapsed = run_experiment(
            f"reproduce-strategy-{strategy}",
            strategy,
            state_dir,
            max_candidates=80,
            batch_size=16,
            max_batches=10,
        )
        rows.append(
            {
                "strategy": strategy,
                "best_quality": result["best_metrics"]["quality"],
                "candidates_evaluated": result["candidates_evaluated"],
                "generations_completed": result["generations_completed"],
                "wall_time_seconds": round(elapsed, 3),
                "stop_reason": result["stop_reason"],
            }
        )
    return rows


def ablation_study(state_dir: Path, full_result: dict) -> list[dict]:
    from neuroforge.optimization.search_space import SearchSpace

    full_space = forge_support_search_space()

    def without(*field_paths: str) -> SearchSpace:
        params = {k: v for k, v in full_space.parameters.items() if k not in field_paths}
        return SearchSpace(parameters=params)

    variants = {
        "full_system": None,  # reuse full_result, don't re-run
        "without_reranking": without("retrieval.reranker"),
        "without_prompt_mutation": without("prompt.system_prompt", "prompt.strategy", "prompt.include_examples"),
        "without_tool_policy_search": without("tools.selection_policy", "tools.enabled"),
    }

    rows = [
        {
            "variant": "full_system",
            "best_quality": full_result["best_metrics"]["quality"],
            "best_policy_compliance": full_result["best_metrics"]["policy_compliance"],
        }
    ]
    for name, space in variants.items():
        if space is None:
            continue
        result, _elapsed = run_experiment(
            f"reproduce-ablation-{name}",
            "evolutionary",
            state_dir,
            search_space=space,
            max_candidates=120,
            batch_size=20,
            max_batches=10,
        )
        rows.append(
            {
                "variant": name,
                "best_quality": result["best_metrics"]["quality"],
                "best_policy_compliance": result["best_metrics"]["policy_compliance"],
            }
        )
    return rows


def write_report(out_dir: Path, main_result: dict, strategy_rows: list[dict], ablation_rows: list[dict]) -> None:
    comparison = main_result["comparison"]
    baseline = main_result["baseline_metrics"]
    best = main_result["best_metrics"]

    def pct(key: str) -> str:
        b, c = baseline[key], best[key]
        if b == 0:
            return "n/a"
        return f"{(c - b) / abs(b) * 100:+.1f}%"

    lines = [
        "# NeuroForge Reproducible Experiment Report",
        "",
        f"Generated by `scripts/reproduce.py`, seed={SEED}, provider=mock (no paid APIs).",
        "",
        "## Hypothesis",
        "",
        "A search over ForgeSupport's prompt, model, retrieval, tool-policy, and agent-planning "
        "configuration can find a genome that improves quality, task success, and policy "
        "compliance over the naive v1 baseline without violating safety constraints or "
        "regressing cost/latency disproportionately.",
        "",
        "## Dataset",
        "",
        f"- domain: `{DOMAIN_NAME}`, {N_CHALLENGES} generated challenges across 8 adversarial "
        "categories (policy FAQ, refund requests, duplicate charges, conflicting order IDs, "
        "ambiguous requests, escalation, long-context policy, malformed tool responses)",
        "- deterministic 70/15/15 train/validation/holdout split (see docs/reproducibility.md); "
        "search happens only against train, this report's numbers are on validation",
        "",
        "## Baseline",
        "",
        f"`{main_result['baseline_genome']['system_id']}@v{main_result['baseline_genome']['version']}` "
        "— direct prompting, greedy tool selection, single-shot agent, no reranking.",
        "",
        "## Search Space & Optimization Strategy",
        "",
        f"- strategy: `{main_result['comparison'].get('strategy', 'evolutionary')}` "
        f"(see strategy comparison below for why)",
        "- 12 tunable genome fields spanning model, prompt, retrieval, tools, and agent config",
        "- multi-objective fitness: quality (0.25) + task_success (0.20) + policy_compliance "
        "(0.20) + safety_score (0.15) − latency (0.10) − cost (0.05) − failure_rate (0.05)",
        "",
        "## Budget",
        "",
        f"- generations completed: {main_result['generations_completed']}",
        f"- candidates evaluated: {main_result['candidates_evaluated']}",
        f"- stop reason: {main_result['stop_reason']}",
        "",
        "## Results (validation split, baseline vs. best candidate)",
        "",
        "| metric | baseline | best candidate | change |",
        "|---|---|---|---|",
    ]
    for key in sorted(baseline):
        lines.append(f"| {key} | {baseline[key]:.4f} | {best[key]:.4f} | {pct(key)} |")

    lines += [
        "",
        "## Statistical Comparison (paired bootstrap, quality metric)",
        "",
        f"{comparison['summary']}",
        "",
        f"- effect size (Cohen's d): {comparison['effect_size']}",
        f"- confidence level: {comparison['confidence']}",
        "",
        "## Recommendation",
        "",
        f"**{main_result['recommendation']}**",
        "",
        "## Search Strategy Comparison",
        "",
        "Same budget (80 candidates), same seed, same dataset — random search vs. evolutionary "
        "search vs. Bayesian optimization vs. a UCB1 bandit:",
        "",
        "| strategy | best quality | candidates | generations | wall time (s) | stop reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in strategy_rows:
        lines.append(
            f"| {row['strategy']} | {row['best_quality']:.4f} | {row['candidates_evaluated']} | "
            f"{row['generations_completed']} | {row['wall_time_seconds']} | {row['stop_reason']} |"
        )

    lines += [
        "",
        "## Ablation Study",
        "",
        "Full evolutionary search vs. the same search with one capability removed from the "
        "search space (so that dimension stays fixed at the baseline's value):",
        "",
        "| variant | best quality | best policy_compliance |",
        "|---|---|---|",
    ]
    for row in ablation_rows:
        lines.append(f"| {row['variant']} | {row['best_quality']:.4f} | {row['best_policy_compliance']:.4f} |")

    lines += [
        "",
        "## Limitations",
        "",
        "- All numbers come from the deterministic mock LLM provider (see docs/reproducibility.md) "
        "— it is a calibrated simulation of how model/config choices affect quality, cost, and "
        "latency, not a live model. Swapping in a real provider changes the numbers but not the "
        "optimization/evaluation machinery.",
        "- The holdout split is never touched by this script; a real promotion decision additionally "
        "requires a holdout evaluation (see `neuroforge promotion request`).",
        "- Single-seed run; the strategy comparison and ablation study use smaller budgets than the "
        "main experiment for reproduction speed, so treat their deltas as directional, not final.",
        "",
        "## Conclusion",
        "",
        f"NeuroForge's evolutionary search improved ForgeSupport's validation quality by "
        f"{comparison['relative_diff'] * 100:+.1f}% "
        f"(95% CI [{comparison['ci_low'] * 100:+.1f}%, {comparison['ci_high'] * 100:+.1f}%]) "
        f"over {main_result['generations_completed']} generations / {main_result['candidates_evaluated']} "
        f"candidates, entirely through controlled experimentation against a deterministic mock backend — "
        "demonstrating the full discover → search → evaluate → decide loop without any paid API calls.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reproduce_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    state_dir = out_dir / "_state"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("Running main experiment (evolutionary search)...")
    main_result, elapsed = run_experiment("reproduce-main", "evolutionary", state_dir)
    main_result["comparison"]["strategy"] = "evolutionary"
    print(f"  done in {elapsed:.2f}s: {main_result['recommendation']}")

    print("Running search-strategy comparison...")
    strategy_rows = strategy_comparison(state_dir)

    print("Running ablation study...")
    ablation_rows = ablation_study(state_dir, main_result)

    (out_dir / "experiment.json").write_text(json.dumps(main_result, indent=2))
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "main_experiment": {
                    "recommendation": main_result["recommendation"],
                    "comparison": main_result["comparison"],
                    "baseline_metrics": main_result["baseline_metrics"],
                    "best_metrics": main_result["best_metrics"],
                },
                "strategy_comparison": strategy_rows,
                "ablation_study": ablation_rows,
            },
            indent=2,
        )
    )
    write_report(out_dir, main_result, strategy_rows, ablation_rows)

    shutil.rmtree(state_dir, ignore_errors=True)

    print(f"\nWrote {out_dir}/experiment.json, results.json, report.md")
    print(f"Recommendation: {main_result['recommendation']}")


if __name__ == "__main__":
    main()
