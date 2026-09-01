# Performance

Measured on the development machine (not a dedicated benchmark environment — treat these as
order-of-magnitude, not SLA numbers) via `python -c` micro-benchmarks against the real code paths,
mock provider, SQLite. Reproduce with the snippets below; nothing here is invented.

| operation | throughput |
|---|---|
| ForgeSupport candidate evaluation (mock provider) | ~10,300 evaluations/sec |
| genome `model_dump(mode="json")` | ~273,000 calls/sec |
| genome content hashing (SHA-256) | ~67,000 hashes/sec |
| mutation candidate generation (`MutationEngine.propose`) | ~13,900 candidates/sec |
| Pareto frontier over 1,000 candidates | ~173ms (O(n²) pairwise dominance check) |
| evolutionary search (`ask`+`tell`, pop=24) | ~6,300 generations/sec (excludes evaluation cost) |
| genome writes to SQLite | ~2,050 writes/sec |

## What this means in practice

Evaluation against the mock provider, not the optimizer bookkeeping, is the bottleneck — a
240-candidate experiment with a 47-challenge train split (the `scripts/reproduce.py` main
experiment) evaluates ~11,000 (genome, challenge) pairs and completes in about a second. A real
LLM provider would make evaluation the bottleneck by orders of magnitude (network latency per
request, not microseconds), which is exactly why the mock provider exists for CI/development and
budget enforcement (`ExperimentBudget.max_requests`/`max_cost_usd`) exists for real-provider runs.

## Known scaling limits

- **Pareto frontier is O(n²)** in the number of candidates compared — fine for the dozens-to-low-
  hundreds of candidates a single experiment produces, would need a sweep-line algorithm
  (O(n log n)) if ever run over tens of thousands of points at once (e.g. cross-experiment
  leaderboards).
- **The bandit strategy used to be combinatorially explosive** — see docs/optimization.md and
  `test_bandit_arm_count_stays_bounded_in_high_dimensional_spaces` — fixed by bounding the arm
  pool via sampling instead of a full Cartesian-product grid.
- **SQLite genome writes (~2,000/sec)** are single-threaded and lock the whole file per write;
  Postgres (the docker-compose/production default) does not share this limitation.

## Reproducing these numbers

```python
import time
from neuroforge.genomes.schema import SystemGenome
from neuroforge.domains import get_domain
from neuroforge.datasets.evolution import seed_dataset
from neuroforge.providers.mock import MockLLMProvider

domain = get_domain("forge-support")
provider = MockLLMProvider()
baseline = SystemGenome(system_id="bench", version=1)
challenges = seed_dataset(domain, "bench-ds", n=500, seed=1).challenges

start = time.perf_counter()
for c in challenges:
    domain.evaluate(baseline, c, provider)
print(len(challenges) / (time.perf_counter() - start), "evaluations/sec")
```
