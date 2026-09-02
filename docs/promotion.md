# Promotion

## Stages

```
GENERATED -> VALIDATED -> BENCHMARKED -> HOLDOUT_TESTED -> APPROVED -> CANARY -> PROMOTED
                                                        \-> REJECTED
                                              CANARY ---> ROLLED_BACK
```

`genomes/schema.py:PromotionStatus` and `promotion/gates.py:PROMOTION_ORDER`/`next_allowed_status`
define the sequence; a genome's `status` field is one of these values. Nothing in this codebase
auto-advances a genome through every stage — each stage requires an explicit call
(`neuroforge promotion request`, `neuroforge canary run`), matching section 69's core distinction:
**autonomous discovery, not autonomous deployment.**

## The human-approval gate (APPROVED/CANARY → PROMOTED)

`PROMOTED` is reachable only through one explicit, attributable action:
`POST /api/v1/promotions/finalize` (`neuroforge promotion promote` on the CLI). It requires, on
record, both:

1. The latest `promotion_decisions` row for the genome has `approved: true`.
2. The latest `canary_runs` row for the genome has `rollback_triggered: false`.

and it requires an **admin**-role API key — `request_promotion` and `run_canary` only need
`operator`, the role automation/CI holds, so a pipeline can run every check up through canary on
its own but cannot itself take a candidate live. The action is logged to `promotion_approvals`
(genome hash, experiment id, the caller's own name, timestamp) — a separate table from
`promotion_decisions` because it records a human decision, not an automated gate result. The
dashboard's `/promotions` page surfaces this as a "Promote to production" button that stays
disabled (with a tooltip explaining why) until both conditions above are met.

## Promotion gates

`promotion/gates.py:evaluate_promotion()` checks, in order:

1. Hard safety constraints (see docs/safety.md) — short-circuits on failure.
2. `min_quality` — absolute floor, independent of the baseline.
3. `max_cost_increase` / `max_latency_increase` — fractional caps vs. baseline.
4. Statistical significance — `comparison.conclusion == LIKELY_IMPROVEMENT` required by default
   (`require_statistically_significant_improvement`); a candidate that merely scores higher on a
   single point estimate is not enough (see docs/evaluation.md / statistics.py).
5. Regression guard — a quality gain paired with a *disproportionate* cost/latency increase
   (more than 2x the configured caps) is rejected even if it technically cleared gates 2–3
   individually.

The result is a `PromotionDecision` with `approved: bool` and a list of human-readable `reasons` —
every rejection states exactly which gate failed and by how much, e.g.
`"policy_violation_rate 0.204 > allowed maximum 0.200"`.

## Why the gates are calibrated the way they are

The defaults (`min_quality=0.62`, `max_policy_violation_rate=0.20`) are calibrated against what
ForgeSupport's deterministic scoring model can actually achieve with a fully-optimized genome
(~0.72–0.79 quality ceiling given the mock provider's capability range and the deliberately
imperfect starting baseline) — set high enough that the naive v1 baseline fails every gate, low
enough that a genuinely improved candidate can clear them. This is the same calibration exercise a
real team does against their own system's achievable range; the numbers are not arbitrary and are
not tuned to force a particular demo outcome — several `scripts/reproduce.py` runs across
different seeds land on both `PROMOTE TO CANARY` and `DO NOT PROMOTE` depending on what the search
actually finds.

## Canary simulation

`promotion/canary.py:simulate_canary()` routes deterministic mock traffic between baseline and
candidate (`candidate_traffic_fraction`, default 10%) via a stable per-challenge hash — so the
same dataset always splits the same way — evaluates both arms independently, and flags
`rollback_triggered` if the candidate's canary-slice safety/quality looks worse than the baseline's
by more than the configured thresholds. This is a second, independent check after search-time
evaluation and the promotion gates, not a rerun of the same evaluation.

## What a real deployment adds

This reference implementation's canary and promotion pipeline are simulations against mock
traffic. A real production version would actually shift live traffic percentages and integrate
with a feature-flag/traffic-routing layer when `finalize_promotion` runs — the human-approval gate
itself (see above) is already implemented here, only the underlying traffic mechanism is
simulated.
