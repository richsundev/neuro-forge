# ADR-0012: Hard safety constraints that override the optimization objective

## Context

A weighted multi-objective fitness function (ADR-0003) can, in principle, promote a candidate
that improved every weighted dimension including a safety-adjacent one, while still crossing an
unacceptable absolute threshold — weighting alone doesn't guarantee a floor.

## Decision

`SafetyConstraints`/`check_safety()` (`promotion/safety.py`) are checked independently of and
prior to the weighted fitness score, both during promotion (`evaluate_promotion` short-circuits on
a safety violation before checking any other gate) and during canary simulation
(`simulate_canary` re-checks safety on live canary-slice data). A candidate cannot be promoted by
having a high enough fitness score to "outweigh" a safety failure — there is no weight that
overrides this gate.

## Alternatives considered

- **Safety as just another weighted objective, with a very high weight.** Rejected: a weight,
  however high, is still a tradeoff the optimizer can trade against given enough gain elsewhere.
  Section 48 of the brief requires an actual override, not a strong preference.

## Tradeoffs

Purely gating on safety (with no reward signal for it during search) means the search has no
incentive to move toward safer regions of the space — it would only ever discover a safe candidate
by chance, then get correctly rejected every other time. This is why `DEFAULT_OBJECTIVES` also
weights `policy_compliance`/`safety_score` in the search objective (ADR-0003) — the gate is the
backstop, not the only mechanism; see docs/safety.md for the concrete tuning history of this
exact issue (an earlier objective weighting reliably found quality gains with policy_compliance
left behind, so the hard gate correctly rejected *every* candidate — safe, but not useful).

## Update (ADR-0014)

The constraint is now enforced statistically: at promotion the policy-violation rate must clear its
limit at a 95% upper confidence bound on the holdout split (a point estimate on a few dozen
challenges passes borderline candidates by luck), and the search itself is constraint-aware, so the
constraint shapes which candidate wins instead of only vetoing it afterwards.

## Consequences

Every promotion rejection names the specific violated constraint and by how much
(`"policy_violation_rate 0.291 (upper 95% bound 0.304) > allowed maximum 0.280"`) — a rejection is never a bare boolean.
