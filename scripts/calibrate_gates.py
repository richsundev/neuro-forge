#!/usr/bin/env python3
"""Calibrate promotion gates and safety limits against a domain's *measured* reachable range.

A limit nobody can satisfy makes every "pass" against it sampling noise; a limit everybody
satisfies gates nothing. This samples the domain's search space, evaluates every sampled
configuration on the whole dataset (population values, not a small split), and reports for each
gated metric how much of the space clears the current defaults — and how much clears *all* of them
at once. Run it after changing a domain's scoring model or before trusting a new domain's limits.

    python scripts/calibrate_gates.py [--domain forge-support] [--n-challenges 300] [--samples 400]
"""

from __future__ import annotations

import argparse
import random
import statistics

from neuroforge.datasets.evolution import seed_dataset
from neuroforge.domains import get_domain
from neuroforge.evaluation.aggregate import aggregate
from neuroforge.experiments.spaces import search_space_for_domain
from neuroforge.genomes.schema import SystemGenome
from neuroforge.promotion.gates import PromotionGateConfig, metric_gate_findings
from neuroforge.promotion.safety import SafetyConstraints
from neuroforge.providers.registry import get_provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="forge-support")
    parser.add_argument("--n-challenges", type=int, default=300)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    domain = get_domain(args.domain)
    provider = get_provider("mock")
    dataset = seed_dataset(domain, "calibration", n=args.n_challenges, seed=1)
    challenges = dataset.challenges
    categories = [c.category for c in challenges]
    gates, limits = PromotionGateConfig(), SafetyConstraints()

    def population(genome: SystemGenome):
        return aggregate([domain.evaluate(genome, c, provider) for c in challenges], categories)

    baseline = SystemGenome(system_id="calibration", version=1)
    baseline_agg = population(baseline)
    space = search_space_for_domain(args.domain)
    rng = random.Random(args.seed)

    rows = []
    for i in range(args.samples):
        genome = baseline.derive(mutations=[], overrides=space.sample(rng), new_version=2 + i)
        agg = population(genome)
        rows.append(
            {
                "quality": agg.metrics["quality"],
                "violation": 1.0 - agg.metrics.get("policy_compliance", 1.0),
                "safety": agg.metrics["safety_score"],
                "cost": agg.cost_usd / baseline_agg.cost_usd - 1.0,
                "latency": agg.latency_ms / baseline_agg.latency_ms - 1.0,
                "clears_all": not metric_gate_findings(baseline_agg, agg, gates, limits),
            }
        )

    def pct(pred) -> str:
        return f"{sum(1 for r in rows if pred(r)) / len(rows):6.1%}"

    print(f"{args.domain}: {args.samples} sampled configurations x {len(challenges)} challenges")
    print(f"baseline: quality {baseline_agg.metrics['quality']:.3f}, "
          f"policy violation {1 - baseline_agg.metrics.get('policy_compliance', 1.0):.3f}\n")
    print(f"{'gate':<34}{'limit':>10}{'cleared by':>12}{'best reachable':>17}")
    table = [
        ("quality >= min_quality", f"{gates.min_quality:.2f}", pct(lambda r: r["quality"] >= gates.min_quality),
         f"{max(r['quality'] for r in rows):.3f}"),
        ("policy violation <= max", f"{limits.max_policy_violation_rate:.2f}",
         pct(lambda r: r["violation"] <= limits.max_policy_violation_rate),
         f"{min(r['violation'] for r in rows):.3f}"),
        ("safety_score >= min", f"{limits.min_safety_score:.2f}", pct(lambda r: r["safety"] >= limits.min_safety_score),
         f"{max(r['safety'] for r in rows):.3f}"),
        ("cost increase <= max", f"{gates.max_cost_increase:+.0%}", pct(lambda r: r["cost"] <= gates.max_cost_increase),
         f"{min(r['cost'] for r in rows):+.0%}"),
        ("latency increase <= max", f"{gates.max_latency_increase:+.0%}",
         pct(lambda r: r["latency"] <= gates.max_latency_increase), f"{min(r['latency'] for r in rows):+.0%}"),
    ]
    for name, limit, cleared, best in table:
        print(f"{name:<34}{limit:>10}{cleared:>12}{best:>17}")
    feasible = [r for r in rows if r["clears_all"]]
    print(f"\nclears ALL gates simultaneously: {len(feasible)}/{len(rows)}")
    if feasible:
        print(f"  best feasible quality {max(r['quality'] for r in feasible):.3f}, "
              f"median {statistics.median(r['quality'] for r in feasible):.3f}")
    else:
        print("  NONE — these limits are jointly unsatisfiable in this search space; every promotion")
        print("  verdict against them would be sampling noise. Recalibrate before trusting them.")


if __name__ == "__main__":
    main()
