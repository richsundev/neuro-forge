# ADR-0016: Experiments you can configure and operate, with cost scored relative to the baseline

## Context

Two things made NeuroForge harder to steer than it looked.

**The objective weights did nothing.** Fitness scored cost and latency against fixed scales (a $0.05
request, a 3-second response). ForgeSupport's baseline costs $0.0012 per request, so those scales
were ~40x too large: a genome three times cheaper gained 0.0008 fitness, while +0.05 quality gained
0.0125. The cost weight was decorative — raising it to 0.4 changed nothing about what the search
returned.

**An experiment could only be created with defaults and run from a terminal.** The search space, the
objective weights and the gates were fixed in code (the API accepted gate overrides but nothing
else), and the dashboard could look at experiments but not launch one.

## Decision

- **Cost and latency are scored relative to the baseline** (`objective_scales` in
  `experiments/engine.py`): each is divided by twice the baseline's own aggregate on the search split,
  so the baseline scores 0.5 and a genome that halves the cost approaches 0.75. Quality-style metrics
  are already in [0, 1] and are untouched. Weights are now a real trade-off: with quality pinned and
  cost weight raised from 0.05 to 0.4, the mean winner's cost across six seeds falls monotonically.
- **Search dimensions are selectable.** `search_dimensions` names sections (`prompt`) or field paths
  (`retrieval.top_k`); everything not listed stays at the baseline's value
  (`experiments/configure.py:build_search_space`). `GET /domains` lists what a domain offers.
- **Objective weights are pinned shares.** `objective_weights={"cost_usd": 0.4}` means cost gets 40%
  of the final total; the unlisted objectives share the remaining 60% in their default proportions
  (`build_objectives`). Weights that already fill the total leave the others at zero. Unknown metrics
  and non-positive weights are a 400, not a silent default.
- **Experiments can be started without blocking.** `POST /experiments/{id}/start` returns 202 and
  runs on an in-process thread pool (bounded, one run per experiment) through the same
  `run_recorded_experiment` the CLI and worker use; `GET /experiments/{id}/checkpoint` reports live progress. This is
  for single-process and demo deployments — production keeps `/enqueue` and the Redis worker
  (ADR-0010).
- **The dashboard launches and follows experiments.** `/experiments/new` chooses the baseline (champion
  by default), the strategy, budget, dimensions, a priority preset (balanced / quality first / cost
  first / latency first) and the gates; the detail page polls and shows live fitness, feasibility and the
  configuration the run actually used. The CLI has the same controls (`--focus`, `--weight`,
  `--min-quality`, `--max-cost-increase`, `--max-latency-increase`, `--max-policy-violation`).

## Consequences

Measured on ForgeSupport (300-challenge dataset, 240 candidates, 10 seeds): 8 searches found a
within-limits winner and 9 were approved on the holdout — the same as before, but the first champion
is stronger (quality 0.671 vs 0.642, +31.1% vs +25.4% over v1 at the same cost), and `make reproduce`
now reports +27.5% (95% CI [+22.0%, +32.9%]) with cost −64.2%.

Tradeoffs:

- **A champion at the ceiling leaves no headroom.** From the new first champion, 10 follow-up searches
  find nothing the holdout review will approve (see ADR-0015): about 0.67 is what this space offers.
  I tried also requiring search candidates to beat the baseline's search-split quality by a margin, to
  stop the optimizer trading quality for cost against an already-tuned champion; it changed nothing
  measurable and was dropped.
- **A scale relative to the baseline moves with the baseline.** The same weights mean "cost relative
  to where you started", so a re-run from a cheaper champion sees a smaller gain per dollar saved.
  That is the intent, but scores are only comparable between experiments that share a baseline.
- **Background runs die with the API process.** `neuroforge experiment resume` picks a
  killed run up from its checkpoint; a durable queue is the Redis worker's job.

`tests/test_experiment_config.py` covers dimension and weight construction, relative cost scoring,
the API validation and background start, and the CLI options.
