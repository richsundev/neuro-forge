# Reproducibility

## `make reproduce`

```bash
make reproduce
```

runs `scripts/reproduce.py`: seeds a ForgeSupport dataset, runs a full evolutionary-search
experiment against the deterministic mock provider, runs a 4-way search-strategy comparison
(random/evolutionary/Bayesian/bandit) at equal budgets, runs a 4-variant ablation study, and writes
`reproduce_output/experiment.json`, `results.json`, and `report.md` — a paper-style report
(Hypothesis / Dataset / Baseline / Search Space / Budget / Results / Statistical Comparison /
Strategy Comparison / Ablation / Limitations / Conclusion). Every number in that report comes from
an actual run of the code in the checkout; nothing is hand-typed or hard-coded.

## What "reproducible" means here, precisely

Given the same seed, the same experiment produces **identical genomes, identical metrics,
identical statistical conclusions, and identical recommendations** — verified by running
`scripts/reproduce.py` twice in independent Python processes with *different*
`PYTHONHASHSEED` values and diffing the output. The only fields that differ between runs are
wall-clock timestamps and wall-clock duration measurements, which are supposed to vary.

## Two real bugs this caught (kept here, not just in git history, because they're instructive)

**1. `hash()` is per-process salted.** The domain evaluators originally derived the mock
provider's per-request seed via `hash((genome.hash(), challenge.challenge_id))`. Python salts
`hash()` on `str`/`tuple` per-process by default (`PYTHONHASHSEED`, a security feature since
Python 3.3) — so the *same* experiment run twice, in two different processes, produced two
different sequences of mock-provider outputs, and therefore two different search trajectories and
two different final recommendations. Fixed by routing every such seed through
`util.stable_int()`, a hashlib-based (not builtin-`hash`-based) deterministic integer. Regression
test: `tests/test_reproducibility.py` runs the same evaluation in two subprocesses with
`PYTHONHASHSEED=1` and `PYTHONHASHSEED=2` and asserts identical output — a same-process test
cannot catch this class of bug, which is exactly why it slipped through initially.

**2. `set()` iteration order leaked into JSON key order.** `evaluation/aggregate.py` built its
metrics dict by iterating a `set()` of metric names — also `PYTHONHASHSEED`-dependent, so two
reproductions of the same experiment produced JSON files with the same *values* in a different
*key order*. Fixed by sorting before iterating. Caught the same way: diffing two independent-process
runs, not by inspection.

The lesson generalizes: **same-process reproducibility tests are not sufficient** for anything that
claims cross-run/cross-process reproducibility. `test_reproducibility.py` is the regression test
for the general class, not just these two instances.

## Checkpoint-based resumability is a related but distinct property

An experiment interrupted (crash, budget hit, manual cancel) and resumed later reaches the *same*
state as an uninterrupted run with the same total budget — because the search strategy is
reconstructed fresh and every historical `tell()` batch is replayed into it in original order (see
docs/experimentation.md). `tests/test_experiment_engine.py::test_resume_reproduces_uninterrupted_run`
verifies this directly: stop at 8 candidates, resume to 24, and diff against a straight run to 24.

## What is and isn't covered

- Covered: genome content, evaluation metrics, statistical conclusions, promotion/canary
  decisions, checkpoint/resume equivalence.
- Not covered / explicitly non-deterministic: timestamps, wall-clock duration measurements, and
  anything downstream of a *real* (non-mock) provider — see `providers/compatible.py`. Swapping in
  a real model trades reproducibility for real-world signal; that tradeoff is the provider
  abstraction's whole point (ADR-0010).
