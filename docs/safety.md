# Safety

## Hard constraints override the optimization objective

`promotion/safety.py:SafetyConstraints` (`min_safety_score`, `max_policy_violation_rate`) are
checked by `check_safety()` independently of the weighted fitness score. A candidate can have a
higher fitness than the baseline on every optimized dimension and still be rejected —
`promotion/gates.py:evaluate_promotion()` checks safety *first* and short-circuits with
`SAFETY VIOLATION: ...` in the rejection reasons regardless of how good everything else looks.
This is the literal implementation of section 48's requirement: "NeuroForge must never optimize
solely for task success if doing so violates safety constraints."

## Safety is also part of the search objective, not only a promotion-time gate

`DEFAULT_OBJECTIVES` (`evaluation/objectives.py`) weights `policy_compliance` (0.20) and
`safety_score` (0.15) directly in the fitness function the search strategies optimize — not just
quality and cost. Earlier iterations of this project weighted only quality/task_success/cost, and
the search reliably found candidates that improved quality while leaving policy_compliance
essentially where the baseline left it (nothing was pulling the search toward it) — the hard gate
would then correctly reject *every* candidate, which is safe but not useful. Baking safety into
the reward signal, with the hard gate as a backstop rather than the only mechanism, is what lets
the search actually find safe-and-good candidates instead of only proving no unsafe one got
promoted.

## Regression guard

`evaluate_promotion()` rejects a candidate whose quality gain is disproportionate to its cost or
latency increase (`> 2x` the configured `max_cost_increase` / `max_latency_increase`) even when
every other gate passes — see docs/promotion.md.

## Canary rollback

`promotion/canary.py:simulate_canary()` re-checks safety constraints and a quality-regression
threshold on the *canary slice specifically* (not the same evaluation the candidate was searched
against) before recommending promotion — `rollback_triggered=True` if the candidate's live-traffic
behavior looks worse than what search-time evaluation predicted.

## What "safety" means in the ForgeSupport domain, concretely

`domains/forge_support.py:_safety_score()` penalizes: a `greedy` tool-selection policy combined
with the refund tool enabled on risky categories (the "agent issued a wrongful refund" failure
mode), weak prompt constraints on risky categories, and free-text output on refund-adjacent flows.
This is a deliberately narrow, concrete definition of safety for one demo domain — a production
system would define domain-specific safety scoring the same way (see `ApplicationDomain.
safety_constraints()` for the plugin point), not adopt this exact formula.
