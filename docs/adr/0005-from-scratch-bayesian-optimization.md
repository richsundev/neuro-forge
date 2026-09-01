# ADR-0005: A from-scratch Gaussian Process instead of a Bayesian-optimization library

## Context

Bayesian optimization is valuable when evaluations are expensive (a real LLM provider, not the
mock) and sample efficiency matters more than wall-clock optimizer overhead. NeuroForge needed to
support it as one of five comparable search strategies.

## Decision

`BayesianSearchStrategy` (`optimization/bayesian.py`) implements a Gaussian Process (RBF kernel,
closed-form posterior via `numpy.linalg.pinv`) with Expected Improvement acquisition, entirely in
NumPy/SciPy — no `scikit-optimize`, `Optuna`, `BoTorch`, or similar.

## Alternatives considered

- **A dedicated Bayesian-optimization library.** More features (multiple acquisition functions,
  better-conditioned GP fitting, batch acquisition), but a heavyweight dependency for something
  this small, and most such libraries assume a purely numeric search space — `SearchSpace`'s
  mixed numeric/categorical/list-valued parameters would need adapter code regardless.

## Tradeoffs

The from-scratch GP uses `pinv` rather than a Cholesky decomposition with jitter, which is less
numerically robust for ill-conditioned kernels at scale — acceptable at the candidate-pool sizes
this system actually uses (hundreds, not tens of thousands), not a general-purpose GP
implementation.

## Consequences

`SearchSpace.encode()` (min-max scaling for numeric, one-hot for categorical) is the one piece of
adapter code needed, shared by nothing else — small enough to read and verify in one sitting,
which matters for a project whose whole premise is that search/evaluation correctness is
inspectable, not a black box. See `test_bayesian_search_handles_list_valued_categoricals` for a
regression this caught (list-valued categoricals broke the pool-deduplication key).
