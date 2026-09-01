# ADR-0002: One Python distribution with subpackages, not nine separate packages

## Context

A natural module breakdown for this system is genomes / mutations / optimization / evaluation /
datasets / promotion / providers / domains / experiments — each a plausible unit of ownership.

## Decision

Implement each as a subpackage of one `neuroforge` distribution (`src/neuroforge/<name>/`), with
one `pyproject.toml`, one version, one test suite, one CI pipeline. The API app
(`apps/api`) is a separate installable package that depends on `neuroforge`, because it is a
genuinely separate deployable artifact with its own dependencies (FastAPI, passlib); the
subpackages above are not independently deployable.

## Alternatives considered

- **Nine separately-versioned PyPI packages**, each with its own `pyproject.toml`, cross-package
  dependency pins, and independent release cadence.

## Tradeoffs

A monorepo-of-one-package can't enforce module boundaries as strictly as separate installable
packages (nothing stops `genomes/` from importing `experiments/` at the Python level the way
missing a dependency would). This repo maintains the boundary by convention and code review
discipline, not tooling — a real cost if the team grows.

## Consequences

No cross-package version-pin management for a single-deployable system, one `pip install -e .`
for local dev, one CI matrix. The module *boundaries* (and the "only import downward" discipline —
`experiments` depends on `optimization`, never the reverse) are preserved even though the
packaging boundary isn't; that's the part that actually mattered for keeping search/evaluation/
promotion logic separable and independently testable (see ADR-0006, ADR-0013).
