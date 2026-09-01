"""Cross-process reproducibility (section 20/59): the same genome + challenge + seed must produce
byte-identical evaluation results in a brand new Python process, not just within one process.

This is a regression test for a real bug found while validating `scripts/reproduce.py`: the
domain evaluators used Python's builtin `hash()` to derive the mock provider's per-request seed,
which is randomly salted per-process (PYTHONHASHSEED) unless explicitly fixed — so two runs of
the identical experiment produced different mock outputs and therefore different optimization
trajectories. Fixed by routing everything through `neuroforge.util.stable_int` (hashlib-based)
instead. Running two subprocesses with *different* PYTHONHASHSEED values is what actually catches
this class of bug — same-process tests cannot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_SNIPPET = """
import json
from neuroforge.datasets.evolution import seed_dataset
from neuroforge.domains import get_domain
from neuroforge.genomes.schema import SystemGenome
from neuroforge.providers.mock import MockLLMProvider

domain = get_domain("forge-support")
dataset = seed_dataset(domain, "repro-check", n=20, seed=7)
genome = SystemGenome(system_id="repro-agent", version=1)
provider = MockLLMProvider()

results = [domain.evaluate(genome, c, provider).metrics for c in dataset.challenges]
print(json.dumps(results))
"""


def _run_in_subprocess(hash_seed: str) -> list[dict[str, float]]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(proc.stdout)


def test_evaluation_is_identical_across_processes_with_different_hash_seeds():
    result_a = _run_in_subprocess("1")
    result_b = _run_in_subprocess("2")
    assert result_a == result_b
    assert len(result_a) == 20
