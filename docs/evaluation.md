# Evaluation Methodology

## The mock provider is a calibrated simulation, not a stub

`providers/mock.py:MockLLMProvider` derives every output (capability signal, tokens, latency,
cost) deterministically from a hash of `(seed, model config, prompt content)` — see
`util.stable_rng`. Three model profiles (`fast-cheap`, `reasoning-small`, `reasoning-large`) have
genuinely different capability/cost/latency tradeoffs; temperature, reasoning effort, and context
size all have real, direction-correct effects. This is what lets the optimizer find real signal in
CI without a paid API key — see docs/reproducibility.md for why this had to be fixed once
(Python's builtin `hash()` is per-process salted and broke exact reproducibility until replaced
with a hashlib-based deterministic hash).

## Domain-specific scoring

`domains/forge_support.py` computes `quality`, `faithfulness`, `task_success`, `tool_success`,
`policy_compliance`, `safety_score` from four independently-computed sub-scores
(`_retrieval_quality`, `_tool_quality`, `_prompt_quality`, `_agent_quality`), each a direct,
inspectable function of genome fields, blended per challenge category with category-specific
weights (a `duplicate_charge` challenge weights tool-use quality heavily; a `policy_faq` challenge
weights retrieval quality heavily). This is what makes the ForgeSupport demo's numbers *earned*
rather than hand-tuned to look good — see the safety-score formula specifically for how a genome
that enables refunds with a `greedy` tool policy gets penalized on risky categories, mirroring the
"agent incorrectly issued a refund" failure mode from real support-agent postmortems.

## Multi-evaluator ensemble

`evaluation/judges.py:evaluate_with_ensemble` runs five "evaluator" perspectives (deterministic,
reference, two LLM-judge stand-ins, an embedding-evaluator stand-in) over the same underlying
score, each with a small deterministic bias/noise profile. When their standard deviation exceeds
`AGREEMENT_UNCERTAINTY_THRESHOLD`, the result is flagged `uncertain` — evaluation disagreement is
surfaced, not silently resolved by picking the most favorable judge.

## Aggregation

`evaluation/aggregate.py:aggregate()` reduces a list of per-challenge `EvaluationResult`s to one
`AggregateMetrics` — mean per metric, mean cost/latency, failure rate, and per-category quality.
Metric keys and category keys are explicitly `sorted()`, not left as `set()` iteration order —
JSON output must be identical across processes for `scripts/reproduce.py` to actually reproduce
(see docs/reproducibility.md).

## Statistical comparison

See docs/promotion.md and `evaluation/statistics.py` — a paired bootstrap over per-challenge score
differences, never a bare point-estimate comparison.
