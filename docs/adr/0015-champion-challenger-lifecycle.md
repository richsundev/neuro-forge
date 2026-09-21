# ADR-0015: A champion/challenger production lifecycle

## Context

NeuroForge's premise is *evolving* an LLM system, but the loop was open: every experiment started
from the application's original v1 genome, whatever had been promoted since. Promoting a candidate
changed one status field and nothing else — the next experiment ignored it, a second promotion
didn't replace the first (both showed as PROMOTED), and nothing could take a bad promotion back. Worse,
a candidate approved against v1 could later be promoted over a better champion it had never been
compared with.

## Decision

Each application has at most one **champion** — the genome in production — and the lifecycle around
it is explicit (`promotion/lifecycle.py`):

- **The approval log is the source of truth.** `promotion_approvals` records every promotion *and*
  every rollback (`action`). The champion and the rollback history are replayed from it as a stack
  (a promotion pushes, a rollback removes the genome it names). `system_genomes.status` is kept in
  step as a cache for the UI, so the two can be stale but never contradict what happened.
- **Experiments evolve from the champion.** `POST /experiments` takes `baseline`: `"champion"`
  (default: the genome in production, else the original), `"original"`, or a specific genome (hash or
  `<system>@v<N>`). The resolved genome is recorded on the experiment (`baseline_hash`), so a run and
  its resumes start from the same place. Candidates are derived from it (lineage edges run from the
  champion) and generation numbers continue from its generation. Because the cost/latency gates and
  the significance test are *relative to the baseline*, a candidate now has to beat what is actually
  running, not the naive v1.
- **Promoting supersedes.** The new champion becomes PROMOTED and the previous one SUPERSEDED (a new
  status), so exactly one genome is ever PROMOTED per application.
- **Rollback is one admin action** (`POST /promotions/rollback`, `neuroforge promotion rollback`):
  the champion becomes ROLLED_BACK and the one it replaced is restored (or, from the first champion,
  production returns to the original baseline).
- **A promotion is only valid against the baseline it was measured on.** `finalize` refuses (409) a
  candidate whose decision was made against a genome other than the current production genome —
  otherwise a candidate approved against v1 could replace a champion it never beat, or a rollback could
  leave stale approvals live.
- Per-experiment gate and safety overrides (`promotion_gates`, `safety_constraints`) are accepted at
  creation: a second iteration from an already-optimized champion often wants different limits than
  the first.

## Alternatives considered

- **Store the champion as a column on the application.** Simpler to read, but a second source of truth
  that can disagree with the approval log; replaying the log keeps history and current state
  consistent by construction.
- **Let stale promotions through with a warning.** Rejected: the human-approval gate exists to prevent
  exactly the silent regression a stale comparison would permit.
- **Auto-promote a candidate that beats the champion.** Rejected by ADR-0013: discovery is autonomous,
  deployment is not.

## Tradeoffs

Replaying the log is O(events per application) on every champion lookup — negligible at the scale of
human-approved production changes. A rollback of the *first* champion returns to "the original
baseline", which is not a stored champion; anything measured against the rolled-back champion becomes
stale and must be re-run. Candidate version numbers stay globally unique per application, so the
champion's version is not always the highest.

## Consequences

`ExperimentConfig.baseline_hash`, `PromotionStatus.SUPERSEDED`, `promotion_approvals.action`
(migration `c5ae26e681f0`), `GET /applications` returns each application's `production`,
`GET /promotions/approvals` returns promotions and rollbacks with system and version, and the
dashboard shows the champion, evolves-from labels, production history and a rollback control.
`tests/test_lifecycle.py` covers the chain, supersession, both rollback steps, stale promotions and
the admin requirement. Measured on ForgeSupport, evolving from a promoted champion: with the champion at quality 0.642,
10 follow-up searches found a further statistically significant gain in 8 cases (+2.3% to +4.8%
relative) and 7 cleared the holdout gates. With the baseline-relative objective of
[ADR-0016](0016-configurable-operable-experiments.md) the *first* champion is already better
(quality 0.671, +31.1% over v1 rather than +25.4%), and 10 follow-ups then find nothing to promote:
none is approved (five inconclusive, four likely regressions on the holdout, one likely improvement
that misses a gate). That is the right outcome rather than a defect — 0.67 is about the ceiling of
this search space (the best follow-up in the earlier run reached 0.673), so what's left is noise,
and the promotion review rejected all of it. The lifecycle's job is to make that visible, not to
manufacture a promotion.
