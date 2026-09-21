# ADR-0014: Verify promotions on the holdout, at confidence bounds, against calibrated limits

## Context

End-to-end validation of the promotion pipeline (running it, not reading it) turned up a cluster
of problems that all had the same effect — a "PROMOTE" verdict carried far less evidence than it
appeared to:

1. **The holdout split was never evaluated.** `HoldoutGuard.evaluate_holdout` existed and was
   tested, but no code path called it; `neuroforge promotion request` re-read the experiment's own
   validation numbers. `PromotionStatus.HOLDOUT_TESTED` was never produced.
2. **The safety limit was unreachable.** Across 600 sampled configurations on a 300-challenge
   population, the lowest policy-violation rate any ForgeSupport genome achieves is 0.216 — the
   default limit was 0.20. Every "pass" was small-sample luck: the validation split held 15
   challenges, so a candidate whose true rate was ~0.23 passed a 0.20 point-estimate check most of
   the time, then failed the canary's independent sample every time (all 9 distinct candidates tried).
3. **The cost and latency gates never fired** in the API/CLI path: aggregates were rebuilt from a
   stored metrics dict with `cost_usd`/`latency_ms` left at zero.
4. **Search ignored the limits.** Best-so-far was plain max-fitness, so the winner sat on (or past)
   the boundary of the limits and passed the later checks on noise.
5. **The canary replayed dataset challenges** including the train split the candidate was
   optimized against, and the `dataset_id` an experiment was created with was ignored at run time.

## Decision

- **Promotion is decided on holdout evidence.** `promotion/review.py:review_promotion` (shared by
  API and CLI) measures the candidate and the experiment baseline on the holdout split through
  `HoldoutGuard.evaluate_holdout`, then applies the experiment's own gates and safety limits to
  those numbers. The measurement is persisted (`holdout_evaluations`, unique per genome + dataset
  version) and reused on repeat requests, so "at most once per candidate" now holds across
  processes — closing the gap ADR-0007 called out.
- **Safety is checked at confidence bounds**, not point estimates: the policy-violation rate must
  be below its limit at the 95% one-sided upper bootstrap bound, the safety score above its floor
  at the lower bound (`promotion/safety.py:SafetyEstimate`). The experiment's own recommendation
  uses the same check on the validation split.
- **Limits are calibrated to the measured reachable range**, reproducibly
  (`scripts/calibrate_gates.py`), and a test fails if no configuration in a domain's search space
  can clear all default gates. The policy-violation default moved from 0.20 to 0.28: a looser
  number, enforced statistically instead of by luck.
- **Search is constraint-aware.** Fitness is divided by `1 + penalty × violation` (violation =
  distance outside the safety limits, quality floor, cost and latency gates), selection is
  feasible-first, the safety limits and quality floor are tightened by a small search margin so the
  winner has headroom, and a fitness plateau does not count as convergence until a feasible
  candidate exists. The gates are one implementation (`metric_gate_findings`) shared by search and
  promotion, so the search optimizes toward what promotion will accept.
- **The canary runs on fresh traffic** (`fresh_traffic`): newly generated requests from the
  dataset's generator and difficulty band, disjoint from every split.
- Experiments record their `dataset_id` and the dataset version they ran against, so the review
  evaluates the holdout of the same dataset version.

## Alternatives considered

- **Keep 0.20 and change the simulated domain until it is reachable.** Rejected: tuning the
  simulation to fit a threshold is the wrong direction. The limit is the configurable part.
- **Apply the search margin to every limit.** Tried; with margins on cost and latency too, only 1 in
  600 sampled configurations was feasible, and cost/latency are near-deterministic functions of the
  configuration, so they don't need the buffer. Margins apply to the noisy metrics only.
- **Reuse the validation numbers for the decision.** Validation is unbiased for the single selected
  genome, but it is also the sample the experiment's own recommendation already consumed, and at
  n≈15–45 it is too small to carry a confidence claim by itself.

## Tradeoffs

The holdout is ~15–25% of the dataset, so its bounds are wide (n=50 here); a promotion that needs
tighter claims needs a larger dataset (the default seed size went from 100 to 300 for this reason).
The 0.28 limit is a property of ForgeSupport's simulated scoring model, not a universal number — a
new domain must be calibrated the same way. The constraint-aware search can still fail to find a
feasible winner within budget (2 of 10 seeds at 240 candidates); it then reports
`DO NOT PROMOTE` with the violated limit rather than promoting the least-bad candidate.

## Consequences

`ExperimentResult` gains `dataset_id`, `dataset_version`, `selected_feasible` and
`validation_safety`; `promotion_decisions` gains an `evidence` column and there is a new
`holdout_evaluations` table (migration `4daa6685abe3`). `tests/test_promotion_integrity.py` pins
each of the five failures above.
