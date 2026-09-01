# ADR-0006: Separate candidate generation from evaluation infrastructure

## Context

Candidate generation (what should we try next?) is a natural place to want LLM assistance —
proposing prompt rewrites, hypothesizing which config changes might help. Evaluation (did it
actually work?) determines real money/promotion decisions and must be trustworthy.

## Decision

Candidate generation (`MutationEngine`, search strategies) and evaluation
(`ApplicationDomain.evaluate`, `evaluation/*`) are separate modules with no dependency from
evaluation back to generation. An LLM may in principle back a mutation operator (the prompt-text
mutators in `mutations/operators.py` are templated today but structurally swappable for an LLM
call — see the module docstring); evaluation never calls an LLM to *judge* whether a candidate is
good in a way that could be gamed by the same kind of model that generated the candidate.

## Alternatives considered

- **Let the generating LLM also score its own candidates.** Rejected outright — this is precisely
  the failure mode section 11 of the original brief warns against: the system optimizing against
  its own proposer's opinion of itself, with no independent check.

## Tradeoffs

Deterministic, rule-based evaluation (as implemented for ForgeSupport/SQLAgent/ResearchAgent) is
less flexible than an LLM judge for genuinely open-ended quality assessment — real domains outside
this reference implementation may need actual LLM judges. `evaluation/judges.py`'s ensemble
pattern (multiple evaluators, disagreement surfaced rather than resolved) is the intended pattern
for adding those without collapsing back into "the generator judges itself."

## Consequences

`ApplicationDomain` (the plugin interface) never imports from `mutations/` or `optimization/`, and
neither of those import evaluation logic beyond calling `domain.evaluate()` as an opaque function.
This is enforced by convention/review, not a build-time boundary (see ADR-0002) — worth stating
plainly rather than implying it's mechanically guaranteed.
