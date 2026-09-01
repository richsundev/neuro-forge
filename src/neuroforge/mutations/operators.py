"""Deterministic mutation operators, grouped by the category they belong to.

Each operator inspects the *current* value of a field on a genome and proposes a small set of
candidate replacement values plus a human-readable rationale. Operators are deliberately
deterministic (seeded RNG only) so experiments are reproducible; an LLM can be swapped in behind
`PROMPT_TEXT_MUTATORS` without touching the search or evaluation code (see docs/design-decisions.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from neuroforge.genomes.schema import SystemGenome

CONSTRAINT_LIBRARY = [
    "Never promise a refund without verifying the order ID against the order lookup tool.",
    "If policy and the customer's claim conflict, defer to policy and explain why.",
    "Always state the specific policy clause used to justify a decision.",
    "When information is missing, ask a clarifying question instead of guessing.",
]

EXAMPLE_LIBRARY = [
    "Example: refund request with mismatched order ID -> ask for the correct order ID before proceeding.",
    "Example: duplicate charge complaint -> look up both transactions before deciding.",
]


@dataclass(frozen=True)
class MutationProposal:
    field_path: str
    mutation_type: str
    new_value: Any
    reason: str


Operator = Callable[[SystemGenome], list[MutationProposal]]


def _prompt_add_constraint(genome: SystemGenome) -> list[MutationProposal]:
    existing = genome.prompt.system_prompt
    out = []
    for constraint in CONSTRAINT_LIBRARY:
        if constraint not in existing:
            out.append(
                MutationProposal(
                    "prompt.system_prompt",
                    "prompt.add_constraint",
                    (existing.rstrip() + "\n- " + constraint).strip(),
                    "Adds an explicit behavioral constraint to reduce policy-violation failures.",
                )
            )
    return out[:2]


def _prompt_simplify(genome: SystemGenome) -> list[MutationProposal]:
    lines = [line for line in genome.prompt.system_prompt.split("\n") if line.strip()]
    if len(lines) <= 2:
        return []
    simplified = "\n".join(lines[: max(2, len(lines) // 2)])
    return [
        MutationProposal(
            "prompt.system_prompt",
            "prompt.simplify",
            simplified,
            "Removes redundant instructions to reduce instruction-following overhead and latency.",
        )
    ]


def _prompt_strategy(genome: SystemGenome) -> list[MutationProposal]:
    options = ["direct", "structured_reasoning", "chain_of_thought", "few_shot", "constrained"]
    return [
        MutationProposal(
            "prompt.strategy",
            "prompt.change_strategy",
            opt,
            f"Switches prompting strategy to '{opt}' to test its effect on quality/latency tradeoff.",
        )
        for opt in options
        if opt != genome.prompt.strategy
    ]


def _prompt_examples(genome: SystemGenome) -> list[MutationProposal]:
    return [
        MutationProposal(
            "prompt.include_examples",
            "prompt.toggle_examples",
            not genome.prompt.include_examples,
            "Toggles few-shot examples to test their effect on tool-call accuracy.",
        )
    ]


def _model_temperature(genome: SystemGenome) -> list[MutationProposal]:
    current = genome.model.temperature
    candidates = sorted({round(max(0.0, current - 0.15), 2), round(min(1.0, current + 0.15), 2)})
    return [
        MutationProposal(
            "model.temperature",
            "model.change_temperature",
            c,
            f"Adjusts temperature from {current} to {c} to trade off determinism vs. diversity.",
        )
        for c in candidates
        if c != current
    ]


def _model_name(genome: SystemGenome) -> list[MutationProposal]:
    options = ["reasoning-small", "reasoning-large", "fast-cheap"]
    return [
        MutationProposal(
            "model.name",
            "model.switch_model",
            opt,
            f"Switches the underlying model to '{opt}' to test cost/quality tradeoff.",
        )
        for opt in options
        if opt != genome.model.name
    ]


def _retrieval_top_k(genome: SystemGenome) -> list[MutationProposal]:
    current = genome.retrieval.top_k
    candidates = sorted({max(1, current - 2), current + 2, current + 4})
    return [
        MutationProposal(
            "retrieval.top_k",
            "retrieval.change_top_k",
            c,
            f"Adjusts retrieved-document count from {current} to {c} to test recall/latency tradeoff.",
        )
        for c in candidates
        if c != current
    ]


def _retrieval_chunking(genome: SystemGenome) -> list[MutationProposal]:
    options = [400, 600, 800, 1000]
    return [
        MutationProposal(
            "retrieval.chunk_size",
            "retrieval.change_chunk_size",
            opt,
            f"Changes chunk size to {opt} to test context-completeness vs. noise tradeoff.",
        )
        for opt in options
        if opt != genome.retrieval.chunk_size
    ]


def _retrieval_reranker(genome: SystemGenome) -> list[MutationProposal]:
    options = ["none", "cross_encoder", "mmr"]
    return [
        MutationProposal(
            "retrieval.reranker",
            "retrieval.change_reranker",
            opt,
            f"Enables '{opt}' reranking to test precision improvements on retrieved context.",
        )
        for opt in options
        if opt != genome.retrieval.reranker
    ]


def _tools_policy(genome: SystemGenome) -> list[MutationProposal]:
    options = ["greedy", "risk_aware", "conservative"]
    return [
        MutationProposal(
            "tools.selection_policy",
            "tools.change_selection_policy",
            opt,
            f"Switches tool-selection policy to '{opt}' to test its effect on tool-call accuracy.",
        )
        for opt in options
        if opt != genome.tools.selection_policy
    ]


def _context_ordering(genome: SystemGenome) -> list[MutationProposal]:
    options = ["relevance", "recency", "source_priority"]
    return [
        MutationProposal(
            "context.ordering",
            "context.change_ordering",
            opt,
            f"Changes document ordering to '{opt}' to test faithfulness improvements.",
        )
        for opt in options
        if opt != genome.context.ordering
    ]


def _context_dedup(genome: SystemGenome) -> list[MutationProposal]:
    return [
        MutationProposal(
            "context.deduplicate",
            "context.toggle_dedup",
            not genome.context.deduplicate,
            "Toggles context deduplication to reduce redundant tokens and confusion.",
        )
    ]


def _agent_planning(genome: SystemGenome) -> list[MutationProposal]:
    options = ["single_shot", "react", "plan_and_execute"]
    return [
        MutationProposal(
            "agent.planning_strategy",
            "agent.change_planning_strategy",
            opt,
            f"Switches planning strategy to '{opt}' to test task-success on multi-step requests.",
        )
        for opt in options
        if opt != genome.agent.planning_strategy
    ]


OPERATORS: dict[str, Operator] = {
    "prompt.add_constraint": _prompt_add_constraint,
    "prompt.simplify": _prompt_simplify,
    "prompt.change_strategy": _prompt_strategy,
    "prompt.toggle_examples": _prompt_examples,
    "model.change_temperature": _model_temperature,
    "model.switch_model": _model_name,
    "retrieval.change_top_k": _retrieval_top_k,
    "retrieval.change_chunk_size": _retrieval_chunking,
    "retrieval.change_reranker": _retrieval_reranker,
    "tools.change_selection_policy": _tools_policy,
    "context.change_ordering": _context_ordering,
    "context.toggle_dedup": _context_dedup,
    "agent.change_planning_strategy": _agent_planning,
}
