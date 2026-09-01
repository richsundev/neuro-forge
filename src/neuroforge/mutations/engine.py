"""MutationEngine: turns a genome + policy into a set of policy-compliant candidate genomes."""

from __future__ import annotations

import random

from neuroforge.genomes.schema import MutationRecord, SystemGenome
from neuroforge.mutations.operators import OPERATORS, MutationProposal
from neuroforge.mutations.policy import MutationPolicy
from neuroforge.util import get_field_by_path


class MutationEngine:
    """Generates candidate genomes by composing 1..N field-level mutations per candidate.

    Candidate generation may in principle be backed by an LLM (see docs/design-decisions.md ADR
    on separating generation from evaluation) but must always be filtered through
    `MutationPolicy` before it is allowed to become an experiment candidate — see
    `MutationPolicy.validate_change_set`.
    """

    def __init__(self, policy: MutationPolicy, seed: int = 0) -> None:
        self.policy = policy
        self._rng = random.Random(seed)

    def propose(
        self,
        genome: SystemGenome,
        n_candidates: int,
        next_version_start: int,
        focus_categories: list[str] | None = None,
    ) -> list[SystemGenome]:
        """Propose up to `n_candidates` policy-compliant children of `genome`."""
        candidates: list[SystemGenome] = []
        op_names = list(OPERATORS.keys())
        if focus_categories:
            focused = [n for n in op_names if any(n.startswith(c) for c in focus_categories)]
            op_names = focused or op_names

        attempts = 0
        seen_signatures: set[tuple] = set()
        while len(candidates) < n_candidates and attempts < n_candidates * 20 + 20:
            attempts += 1
            n_changes = self._rng.randint(1, self.policy.max_changes_per_candidate)
            chosen_ops = self._rng.sample(op_names, k=min(n_changes, len(op_names)))

            proposals: list[MutationProposal] = []
            for op_name in chosen_ops:
                options = OPERATORS[op_name](genome)
                options = [o for o in options if self.policy.is_allowed(o.field_path)]
                if options:
                    proposals.append(self._rng.choice(options))

            if not proposals:
                continue

            field_paths = [p.field_path for p in proposals]
            ok, _reason = self.policy.validate_change_set(field_paths)
            if not ok:
                continue

            signature = tuple(sorted((p.field_path, str(p.new_value)) for p in proposals))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            overrides = {p.field_path: p.new_value for p in proposals}
            mutation_records = [
                MutationRecord(
                    mutation_type=p.mutation_type,
                    field_path=p.field_path,
                    old_value=get_field_by_path(genome, p.field_path),
                    new_value=p.new_value,
                    reason=p.reason,
                )
                for p in proposals
            ]
            child = genome.derive(
                mutations=mutation_records,
                overrides=overrides,
                new_version=next_version_start + len(candidates),
            )
            candidates.append(child)

        return candidates

    def rationale(self, genome: SystemGenome) -> str:
        """Structured, evidence-based explanation of why this genome differs from its parent."""
        if not genome.mutations:
            return "Baseline genome — no mutations applied."
        lines = [f"Mutation rationale for {genome.short_id()} (parent {genome.parent_hash}):"]
        for m in genome.mutations:
            lines.append(f"- {m.field_path}: {m.old_value!r} -> {m.new_value!r}. {m.reason}")
        return "\n".join(lines)
