# Promotion

## Stages

```
GENERATED -> VALIDATED -> BENCHMARKED -> HOLDOUT_TESTED -> APPROVED -> CANARY -> PROMOTED
                                                        \-> REJECTED         |
                                              CANARY ---> ROLLED_BACK   +-> SUPERSEDED (replaced by a newer champion)
                                                                        +-> ROLLED_BACK (admin rollback)
```

`genomes/schema.py:PromotionStatus` and `promotion/gates.py:PROMOTION_ORDER`/`next_allowed_status`
define the sequence; a genome's `status` field is one of these values. Nothing in this codebase
auto-advances a genome through every stage — each stage requires an explicit call
(`neuroforge promotion request`, `neuroforge canary run`, `neuroforge promotion promote`, `neuroforge promotion rollback`), matching section 69's core distinction:
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

## The champion lifecycle (ADR-0015)

Each application has at most one **champion**, the genome in production:

- **Experiments evolve from it.** `POST /experiments` takes `baseline` — `"champion"` (default; the
  genome in production, else the original), `"original"`, or a genome hash / `<system>@v<N>` — and
  records the resolved genome on the experiment. Candidates are derived from it, and because the
  cost/latency gates and the significance test are relative to the baseline, a candidate has to beat
  what is actually running.
- **Promoting supersedes.** The promoted genome becomes PROMOTED and the previous champion SUPERSEDED,
  so exactly one genome per application is PROMOTED.
- **Rolling back** (`POST /promotions/rollback`, `neuroforge promotion rollback <system>`; admin-only)
  marks the champion ROLLED_BACK and restores the one it replaced — or the original baseline, if it
  was the first.
- **Stale promotions are refused (409).** A candidate is only better than what it was compared with:
  if its decision was made against a genome other than the current production genome, promoting it
  could replace a champion it never beat. Re-run the experiment from the current champion.

Production changes are **serialized per application** (`lifecycle.production_lock`: an advisory `flock`
on the state directory around the whole read-check-write transaction, plus a row lock on Postgres).
Without it, six candidates validated against the same baseline and promoted at once each read "nothing
is in production yet" and all six were promoted; now exactly one succeeds and the rest are refused as
stale.

The approval log (`promotion_approvals`, one row per promotion or rollback) is the source of truth;
the champion and rollback history are replayed from it, and `system_genomes.status` is a cache for
the UI, re-derived from the log after every change (`reconcile_production_statuses`), so a genome that has
been in production keeps its SUPERSEDED/PROMOTED/ROLLED_BACK status even if it is reviewed again.

## Promotion review: what the decision is based on

`POST /api/v1/promotions` / `neuroforge promotion request` run `promotion/review.py:
review_promotion`, which is the only place a promotion decision is made:

1. The candidate and the experiment's baseline are measured on the dataset's **holdout split** —
   the split neither the search (train) nor the experiment's own recommendation (validation) ever
   used — through `HoldoutGuard.evaluate_holdout`. The dataset is the one the experiment actually
   ran against (`ExperimentResult.dataset_id` / `dataset_version`).
2. That measurement is **persisted once** per (genome, dataset version) in `holdout_evaluations`.
   A repeat request reuses it rather than re-measuring, so the holdout can't be re-rolled to chase a
   better number (ADR-0007, ADR-0014).
3. The gates and safety limits **from the experiment's own config** are applied to it, and the
   decision is stored with the evidence that produced it (`promotion_decisions.evidence`).

## Promotion gates

`promotion/gates.py:evaluate_promotion()` applies, in order:

1. Hard safety constraints (see docs/safety.md), checked at **95% confidence bounds** on the holdout
   split — the policy-violation rate's upper bound must be under its limit, the safety score's
   lower bound above its floor.
2. `min_quality` — absolute floor, independent of the baseline.
3. `max_cost_increase` / `max_latency_increase` — fractional caps vs. baseline.
4. Statistical significance — `comparison.conclusion == LIKELY_IMPROVEMENT` required by default
   (`require_statistically_significant_improvement`); a candidate that merely scores higher on a
   single point estimate is not enough (see docs/evaluation.md / statistics.py).
5. Regression guard — a quality gain paired with a *disproportionate* cost/latency increase
   (more than 2x the configured caps) is rejected even if it technically cleared gates 2–3
   individually.

The result is a `PromotionDecision` with `approved`, human-readable `reasons` (every rejection
states exactly which gate failed and by how much, e.g. `"quality 0.6196 below minimum 0.6200"`) and
`evidence` (the holdout sample size, the quality comparison, cost/latency deltas, and the safety
bounds). Checks 1–3 live in `metric_gate_findings`, which the experiment engine also uses while
searching — see below.

## Search and promotion share the same limits

`ExperimentEngine` doesn't just maximize fitness and hope: each candidate's fitness is divided by
`1 + constraint_penalty × violation`, where violation is how far it sits outside the safety limits,
the quality floor and the cost/latency gates (the same `metric_gate_findings` promotion uses), and
selection is **feasible-first** — a candidate inside all limits always beats one outside them.
The safety limits and quality floor are tightened by a small search margin (`search_safety_margin`
0.03, `search_quality_margin` 0.02) because a cost-weighted objective's optimum sits *on* a floor,
and a winner on the boundary passes or fails the holdout on noise. A fitness plateau isn't treated
as convergence until a feasible candidate exists. Measured on ForgeSupport (300-challenge dataset,
240 candidates, 10 seeds): 8 seeds found a feasible winner on the search split and 9 were approved on the holdout
(one search-infeasible winner still cleared it); the other seed was correctly reported as
`DO NOT PROMOTE` with the violated limit (policy-violation upper bound 0.304 against 0.28).

## Why the gates are calibrated the way they are

`python scripts/calibrate_gates.py` samples the search space, evaluates each configuration on the
whole dataset, and reports how much of the space clears each limit and how much clears all of them:

```
gate                                   limit  cleared by   best reachable
quality >= min_quality                  0.62       53.5%            0.782
policy violation <= max                 0.28       10.5%            0.216
safety_score >= min                     0.85       97.8%            0.939
cost increase <= max                    +10%       44.0%             -80%
latency increase <= max                 +15%       54.0%             -75%

clears ALL gates simultaneously: 8/400
```

The baseline fails the quality and policy-violation gates; ~2% of the space clears everything, so
the limits discriminate without being unsatisfiable. That last property is enforced by a test
(`test_default_gates_are_reachable_in_forge_support`). The policy-violation limit was originally
0.20, which no ForgeSupport configuration can meet (the floor is 0.216) — see ADR-0014 for what that
did to earlier "PROMOTE" verdicts. The numbers are properties of ForgeSupport's simulated scoring
model; a different domain must be calibrated the same way.

## Canary simulation

`promotion/canary.py:simulate_canary()` routes traffic between baseline and candidate
(`candidate_traffic_fraction`, default 10%) via a stable per-request hash, evaluates both arms
independently, and flags `rollback_triggered` if the candidate's canary-slice safety/quality looks
worse than the configured thresholds. The traffic is **fresh** (`fresh_traffic`): newly generated
requests from the dataset's generator and difficulty band, disjoint from train, validation and
holdout — a canary that replayed dataset challenges would re-test the candidate on data it was
optimized against. The canary's candidate arm is small (~20 requests at the defaults), so it is a
point-estimate smoke test on a live slice, not a substitute for the holdout's confidence bounds.

## What a real deployment adds

This reference implementation's canary and promotion pipeline are simulations against mock
traffic. A real production version would actually shift live traffic percentages and integrate
with a feature-flag/traffic-routing layer when `finalize_promotion` runs — the human-approval gate
itself (see above) is already implemented here, only the underlying traffic mechanism is
simulated.
