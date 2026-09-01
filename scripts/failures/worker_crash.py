#!/usr/bin/env python3
"""Failure injection: kill a running experiment process with SIGKILL mid-run and verify the
checkpoint lets a fresh process resume rather than restart from zero (section 51/35).

    python scripts/failures/worker_crash.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path("/tmp/neuroforge_failure_worker_crash")
EXPERIMENT_ID = "failure-worker-crash"

RUN_SNIPPET = """
import sys
from pathlib import Path
from neuroforge.datasets.evolution import seed_dataset
from neuroforge.domains import get_domain
from neuroforge.experiments.engine import ExperimentConfig, ExperimentEngine
from neuroforge.experiments.budget import ExperimentBudget
from neuroforge.experiments.spaces import forge_support_search_space
from neuroforge.genomes.schema import SystemGenome

domain = get_domain("forge-support")
dataset = seed_dataset(domain, "failure-crash-ds", n=150, seed=1)
baseline = SystemGenome(system_id="failure-crash-agent", version=1)
config = ExperimentConfig(
    experiment_id="{experiment_id}",
    domain_name="forge-support",
    search_strategy="evolutionary",
    search_space=forge_support_search_space(),
    batch_size=16,
    max_batches=200,
    seed=5,
    budget=ExperimentBudget(max_candidates=400, max_requests=10_000_000, max_cost_usd=1000, max_duration_minutes=30),
)
engine = ExperimentEngine(config, baseline, dataset, Path("{state_dir}") / "{experiment_id}")
result = engine.run()
print("FINISHED", result.candidates_evaluated, result.stop_reason)
"""


def run_worker_process() -> subprocess.Popen:
    snippet = RUN_SNIPPET.format(experiment_id=EXPERIMENT_ID, state_dir=STATE_DIR)
    return subprocess.Popen([sys.executable, "-c", snippet], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def checkpoint_state() -> dict:
    path = STATE_DIR / EXPERIMENT_ID / f"{EXPERIMENT_ID}.checkpoint.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    shutil.rmtree(STATE_DIR, ignore_errors=True)

    print("1. Starting experiment process...")
    proc = run_worker_process()
    time.sleep(1.5)

    print("2. SIGKILL-ing it mid-run (simulating a worker crash)...")
    proc.kill()
    proc.wait()

    before = checkpoint_state()
    candidates_before = sum(len(b["points"]) for b in before.get("tell_batches", []))
    print(f"   checkpoint after crash: {candidates_before} candidates completed, status={before.get('status')}")
    assert before.get("status") == "running", "checkpoint should still show 'running' — the process never reached its normal completion path"
    assert candidates_before > 0, "expected at least one completed batch to have been checkpointed before the crash"

    print("3. Starting a fresh process pointed at the same experiment_id/state_dir (simulating worker restart)...")
    proc2 = run_worker_process()
    stdout, _ = proc2.communicate(timeout=30)
    print("  ", stdout.strip().splitlines()[-1] if stdout.strip() else "(no output)")

    after = checkpoint_state()
    candidates_after = sum(len(b["points"]) for b in after.get("tell_batches", []))
    print(f"   checkpoint after resume: {candidates_after} candidates completed, status={after.get('status')}")

    assert after.get("status") == "completed", "resumed run should reach completion"
    assert candidates_after > candidates_before, "resumed run should have made progress beyond the crash point, not restarted from zero"

    shutil.rmtree(STATE_DIR, ignore_errors=True)
    print(f"\nPASS: recovered from a killed worker via checkpoint/resume ({candidates_before} -> {candidates_after} candidates)")


if __name__ == "__main__":
    main()
