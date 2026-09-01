# Design Decisions

Detailed rationale lives in `docs/adr/` as individual Architectural Decision Records (Context /
Decision / Alternatives / Tradeoffs / Consequences each). This page is the index; read the ADR
itself for the reasoning.

| ADR | Decision |
|---|---|
| [0001](adr/0001-immutable-system-genome.md) | Immutable, content-hashed `SystemGenome` |
| [0002](adr/0002-single-package-not-nine.md) | One Python distribution, subpackages not nine packages |
| [0003](adr/0003-multi-objective-optimization.md) | Weighted multi-objective fitness + separate Pareto view |
| [0004](adr/0004-evolutionary-search-default.md) | Evolutionary search as the default strategy |
| [0005](adr/0005-from-scratch-bayesian-optimization.md) | From-scratch Gaussian Process, not a BO library |
| [0006](adr/0006-separate-search-from-evaluation.md) | Candidate generation never judges its own candidates |
| [0007](adr/0007-holdout-protection.md) | `HoldoutGuard` enforces holdout protection in code |
| [0008](adr/0008-checkpoint-replay-tell-batches.md) | Checkpoint by replaying `tell()` batches |
| [0009](adr/0009-postgres-json-columns.md) | Postgres with JSON columns for nested domain objects |
| [0010](adr/0010-redis-queue-not-celery.md) | A plain Redis queue instead of Celery |
| [0011](adr/0011-deterministic-mock-provider.md) | Deterministic mock LLM provider as default/CI backend |
| [0012](adr/0012-hard-safety-constraints.md) | Hard safety constraints override the fitness score |
| [0013](adr/0013-generation-separate-from-promotion.md) | Generation is fully separate from promotion |

## Bugs found and fixed during development (kept here deliberately)

These aren't hidden in git history because they're informative about what "verify, don't assume"
actually caught in this codebase:

1. **Redis client/server timeout race** crashed the worker on every empty poll
   (`experiments/queue.py` — found by actually running `docker compose up` and reading the logs,
   not by code review).
2. **`hash()` is per-process salted**, breaking cross-run reproducibility of the mock provider
   (found by running `scripts/reproduce.py` twice with different `PYTHONHASHSEED` — see
   docs/reproducibility.md).
3. **`set()` iteration order** leaked into JSON key ordering for the same reason.
4. **Bandit strategy's full Cartesian-product grid** exploded combinatorially on a realistic
   genome search space (millions of arms) — found by noticing a 44-second outlier in
   `scripts/reproduce.py`'s strategy-comparison table.
5. **Recharts entrance animations froze mid-draw** in headless Chrome (page-visibility-gated
   `requestAnimationFrame` throttling), making the dashboard's fitness/Pareto charts render a tiny
   fragment of the real data — found by actually screenshotting the dashboard, not by inspecting
   the component code.

Each has a regression test or is otherwise structurally prevented from recurring — see the
referenced files/tests.
