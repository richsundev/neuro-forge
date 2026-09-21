# Datasets & Challenge Evolution

## Versioning

`datasets/dataset.py:DatasetVersion` is an immutable, hashed snapshot of `Challenge`s (the hash covers
the dataset id and version as well as the challenges — with a content-only hash, two datasets seeded
with the same parameters collided on the database's unique hash and the second was never created),
carrying `parent_version`, `source` (`seed` | `failure_driven` | `benchmark_evolution`), and a
`difficulty_score`. `seed_dataset()` builds the first version; `evolve_if_saturated()` builds the
next one when candidates start saturating the current one.

## Holdout protection — the anti-overfitting mechanism

`datasets/holdout.py:HoldoutGuard` is the actual enforcement point, not just a documented
convention:

- `search_set()` — train split only. This is the only split `ExperimentEngine`'s search loop ever
  touches.
- `validation_set()` — used once per experiment, after search finishes, to pick the final
  candidate and run the statistical comparison.
- `evaluate_holdout(genome_hash)` — the *only* way to reach the holdout split, and it raises
  `HoldoutViolation` if called twice for the same genome hash. Re-running a holdout evaluation to
  chase a better number would defeat the entire point of having one. It is called by the promotion
  review (`promotion/review.py`), never by the experiment engine (a test pins this), and the one
  measurement per (genome, dataset version) is persisted in `holdout_evaluations` so the
  "once" holds across processes and repeated requests, not just within one guard instance.

Split sizes matter: the confidence bounds behind safety checks need a few dozen challenges per
split. `seed_dataset` defaults to 300 challenges (the CLI, API and reproduce script all use it),
which realizes roughly 205 / 45 / 50 train / validation / holdout — at 100 challenges the
validation and holdout splits held 15–25, too few to bound a rate.

`deterministic_split()` assigns each *new* challenge to train/validation/holdout via a stable hash of
`(seed, challenge_id)` — the same challenge always lands in the same split even as the dataset
grows, so adding challenges later doesn't reshuffle earlier promotion decisions' evidence.

## Benchmark evolution

`datasets/evolution.py:evolve_if_saturated()` checks whether the mean search-split score has
crossed `SATURATION_THRESHOLD` (0.90); if so, it generates a new, harder `DatasetVersion` (higher
difficulty band, more challenges) with `parent_version` set — the Challenge Evolution dashboard
page plots `difficulty_score` across versions. This exists so a search can't "win" by exhausting a
static, eventually-too-easy benchmark; see section 63/14 of the original brief for the narrative
this implements.

## Failure-driven challenge generation

`failure_driven_challenges()` generates new challenges concentrated on categories a candidate is
currently failing, rather than uniformly across all categories — turning an observed failure mode
into targeted evaluation material (mirrors the "agent incorrectly issued refund → generate more
refund-edge-case challenges" example from the brief). It returns exactly `n` challenges of the
requested categories (at the hard end of the difficulty range) and fails loudly for a category the
domain doesn't have. `evolve_from_failures()` turns that into a new dataset version
(`source="failure_driven"`) with the same guarantees as saturation-driven evolution: existing
challenges keep their splits and new ids are unique.

It is exposed as `POST /api/v1/datasets/{id}/evolve` with `failing_categories` (no `mean_score`
needed) and `neuroforge dataset evolve --failing-category <name>` (repeatable). Choosing the categories
is left to the caller — read them off an experiment's per-category quality — rather than inferred
automatically.

## Adversarial categories (ForgeSupport)

`policy_faq`, `refund_request`, `duplicate_charge`, `conflicting_order_ids`, `ambiguous_request`,
`escalation_case`, `long_context_policy`, `malformed_tool_response` — each with its own template
family and its own weighting of which genome sub-scores matter (see docs/evaluation.md).
