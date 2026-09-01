# ADR-0001: Represent applications as immutable, content-hashed SystemGenomes

## Context

NeuroForge needs to compare, version, and trace the lineage of LLM application configurations
(prompt, model, retrieval, tools, agent behavior, output format) across an unbounded search
process. Configurations must be safely shared across candidates, comparable for equality, and
traceable back to the experiment that produced them.

## Decision

`SystemGenome` (`genomes/schema.py`) is a frozen Pydantic model. Every mutation produces a new
genome via `.derive()`, which copies the parent's fields, applies overrides, and computes a new
SHA-256 content hash (`hash()`) over everything except metadata (version, timestamps, status).
Identical configuration always hashes identically, regardless of how many times or which path
produced it.

## Alternatives considered

- **Mutable config objects with a separate version table.** Rejected: makes "is candidate X the
  same as one we already tried" an explicit query instead of an equality check, and makes
  accidental cross-candidate mutation a real risk during concurrent search.
- **UUID-based versioning instead of content hashing.** Rejected: two searches that happen to
  reach the same configuration from different paths would get different IDs, hiding a real
  duplicate-work signal (`GenomeStore.add` deduplicates by hash for free).

## Tradeoffs

Content hashing means the hash must exclude anything non-deterministic (timestamps) or the same
logical config would hash differently — `content_dict()` explicitly whitelists the fields that
participate in identity, which is one more thing to keep in sync when the schema grows.

## Consequences

Genome equality, deduplication, and lineage graphs (`GenomeStore.lineage_graph`, the dashboard's
Evolution Graph) all fall out of hashing + parent-pointer for free rather than needing bespoke
comparison logic. Every search strategy, regardless of algorithm, produces the same kind of
first-class, hashed, lineaged object (see ADR-0006).
