# ADR-0011: A deterministic mock LLM provider as the default and CI backend

## Context

The entire platform's value proposition — automated search finding real improvements, verified
statistically — has to be demonstrable without a paid API key, and has to run in CI without
network access or nondeterministic external responses.

## Decision

`providers/mock.py:MockLLMProvider` derives capability signal, token counts, latency, and cost
deterministically from a hash of `(seed, model config, prompt content)`, with three model
profiles at genuinely different capability/cost/latency points and direction-correct effects for
temperature, reasoning effort, and context size. `LLMProvider` (`providers/base.py`) is the
abstraction both this and real providers (`providers/compatible.py`) implement, so the
optimization/evaluation code never knows which one it's talking to.

## Alternatives considered

- **Record-and-replay against a real provider** (VCR-style cassettes). Would give more "realistic"
  text, but ties CI to a fixed snapshot of one model's behavior at one point in time, doesn't
  scale to the combinatorial space of genome configurations the optimizer explores (every distinct
  prompt/model/temperature combination would need its own recorded cassette), and doesn't let the
  optimizer explore configurations nobody thought to record ahead of time.
- **Return fixed/random scores unrelated to configuration.** Would make the demo fast but
  meaningless — the search would have nothing real to find, and any observed "improvement" would
  be noise, not signal. Explicitly the failure mode section 64 of the brief warns against
  ("fabricated benchmark metrics").

## Tradeoffs

The mock provider is a *calibrated simulation*, not a real model — its numbers are internally
consistent and direction-correct by construction (this system's own design choices), not validated
against real-world LLM behavior. Swapping to `providers/compatible.py` for real signal is the
explicit next step for any user of this reference implementation, and is a config change
(`ModelConfig.provider`), not a code change (ADR-0006's provider abstraction is what makes this
true).

## Consequences

This is also what caught a real bug: the mock provider's determinism depends on not using Python's
per-process-salted builtin `hash()` anywhere in the seed derivation — see docs/reproducibility.md
for the bug this actually was, and `util.stable_int()` for the fix.
