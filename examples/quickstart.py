#!/usr/bin/env python3
"""Minimal example: use NeuroForge's Python API directly (no CLI, no HTTP, no database) to run
one small optimization experiment against ForgeSupport and print the result.

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from neuroforge.datasets.evolution import seed_dataset
from neuroforge.domains import get_domain
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
from neuroforge.experiments.spaces import forge_support_search_space
from neuroforge.genomes.schema import SystemGenome

domain = get_domain("forge-support")
dataset = seed_dataset(domain, "quickstart-dataset", n=60, seed=1)
baseline = SystemGenome(system_id="quickstart-agent", version=1)

config = ExperimentConfig(
    experiment_id="quickstart-1",
    domain_name="forge-support",
    search_strategy="evolutionary",
    search_space=forge_support_search_space(),
    batch_size=12,
    max_batches=8,
    seed=1,
    budget=ExperimentBudget(max_candidates=60, max_requests=100_000, max_cost_usd=10, max_duration_minutes=5),
)

engine = ExperimentEngine(config, baseline, dataset, Path("/tmp/neuroforge-quickstart"))
result = engine.run()

print(f"status: {result.status} ({result.stop_reason})")
print(f"generations: {result.generations_completed}, candidates: {result.candidates_evaluated}")
print(f"baseline quality:  {result.baseline_metrics['quality']:.3f}")
print(f"best candidate quality: {result.best_metrics['quality']:.3f}")
print(f"comparison: {result.comparison['summary']}")
print(f"recommendation: {result.recommendation}")
