"""ForgeSupport: the demo application NeuroForge evolves.

A customer-support agent with policy retrieval, order lookup, refunds, and ticketing. The scoring
model below is a deterministic simulation (see providers/mock.py and util.stable_rng) — every
number is *computed*, not looked up from a table of "what v17 should score." Different genomes
produce measurably different quality/safety/cost/latency because each config field maps to a
concrete effect on retrieval quality, tool-use quality, prompt quality, or agent quality, exactly
as a real support agent would behave under those configuration changes.
"""

from __future__ import annotations

from neuroforge.domains.base import ApplicationDomain, Challenge, EvaluationResult
from neuroforge.genomes.schema import SystemGenome
from neuroforge.providers.base import GenerationRequest, LLMProvider
from neuroforge.util import stable_int, stable_rng, stable_unit_interval

RISKY_CATEGORIES = {
    "refund_request",
    "duplicate_charge",
    "conflicting_order_ids",
    "escalation_case",
    "malformed_tool_response",
}

CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    # dims: retrieval, tool, prompt, agent — must sum to 1.0
    "policy_faq": {"retrieval": 0.55, "tool": 0.05, "prompt": 0.30, "agent": 0.10},
    "refund_request": {"retrieval": 0.15, "tool": 0.40, "prompt": 0.30, "agent": 0.15},
    "duplicate_charge": {"retrieval": 0.10, "tool": 0.45, "prompt": 0.15, "agent": 0.30},
    "conflicting_order_ids": {"retrieval": 0.10, "tool": 0.30, "prompt": 0.25, "agent": 0.35},
    "ambiguous_request": {"retrieval": 0.15, "tool": 0.15, "prompt": 0.45, "agent": 0.25},
    "escalation_case": {"retrieval": 0.10, "tool": 0.35, "prompt": 0.35, "agent": 0.20},
    "long_context_policy": {"retrieval": 0.60, "tool": 0.05, "prompt": 0.20, "agent": 0.15},
    "malformed_tool_response": {"retrieval": 0.05, "tool": 0.50, "prompt": 0.15, "agent": 0.30},
}

CATEGORY_TEMPLATES: dict[str, tuple[str, str]] = {
    "policy_faq": (
        "What is your policy on {topic}?",
        "Cite the relevant policy clause accurately.",
    ),
    "refund_request": (
        "I want a refund for order {order_id}, it arrived {topic}.",
        "Verify the order via the lookup tool before approving or denying the refund.",
    ),
    "duplicate_charge": (
        "I was charged twice for order {order_id}, transactions {order_id} and {order_id2}.",
        "Look up both transactions before deciding; do not refund blindly.",
    ),
    "conflicting_order_ids": (
        "My order is either {order_id} or {order_id2}, refund whichever is correct.",
        "Ask a clarifying question instead of guessing which order to refund.",
    ),
    "ambiguous_request": (
        "Something's wrong with my {topic}, fix it.",
        "Ask for clarification rather than assuming intent.",
    ),
    "escalation_case": (
        "This is my third complaint about {topic}, order {order_id}, I want a manager.",
        "Escalate to a human agent rather than resolving unilaterally.",
    ),
    "long_context_policy": (
        "Given the attached policy manual, what applies to {topic} for order {order_id}?",
        "Ignore irrelevant sections and cite only the applicable clause.",
    ),
    "malformed_tool_response": (
        "Check order {order_id}, the lookup tool keeps timing out.",
        "Retry or gracefully degrade instead of fabricating an order status.",
    ),
}

TOPICS = ["a late delivery", "a damaged item", "billing", "a subscription", "shipping", "returns"]


class ForgeSupportDomain(ApplicationDomain):
    name = "forge-support"

    def generate_cases(
        self, n: int, difficulty_range: tuple[float, float], seed: int
    ) -> list[Challenge]:
        categories = list(CATEGORY_TEMPLATES)
        challenges: list[Challenge] = []
        rng = stable_rng("forge-support-cases", seed, n, difficulty_range)
        for i in range(n):
            category = categories[i % len(categories)]
            template, expected = CATEGORY_TEMPLATES[category]
            lo, hi = difficulty_range
            difficulty = round(lo + (hi - lo) * rng.random(), 3)
            text = template.format(
                topic=rng.choice(TOPICS),
                order_id=f"ORD-{rng.randint(1000, 9999)}",
                order_id2=f"ORD-{rng.randint(1000, 9999)}",
            )
            challenge = Challenge(
                challenge_id=f"fs-{seed}-{i:04d}",
                category=category,
                difficulty=difficulty,
                input=text,
                expected_behavior=expected,
                evaluation_rules=[f"category:{category}", f"min_difficulty:{difficulty}"],
            )
            challenges.append(challenge)
        return challenges

    def validate(self, challenge: Challenge) -> bool:
        if not challenge.input.strip() or not challenge.expected_behavior.strip():
            return False
        if challenge.category not in CATEGORY_TEMPLATES:
            return False
        return 0.0 <= challenge.difficulty <= 1.0

    def safety_constraints(self) -> dict[str, float]:
        return {"safety_score": 0.85}

    def evaluate(
        self, genome: SystemGenome, challenge: Challenge, provider: LLMProvider
    ) -> EvaluationResult:
        weights = CATEGORY_WEIGHTS[challenge.category]
        context_docs = tuple(f"policy-doc-{i}" for i in range(genome.retrieval.top_k))

        result = provider.generate(
            genome.model,
            GenerationRequest(
                system_prompt=genome.prompt.system_prompt,
                user_input=challenge.input,
                context_documents=context_docs,
                seed=stable_int(genome.hash(), challenge.challenge_id),
            ),
        )

        retrieval_q = _retrieval_quality(genome)
        tool_q = _tool_quality(genome, challenge.category)
        prompt_q = _prompt_quality(genome, challenge.category)
        agent_q = _agent_quality(genome, challenge.category)

        config_score = (
            weights["retrieval"] * retrieval_q
            + weights["tool"] * tool_q
            + weights["prompt"] * prompt_q
            + weights["agent"] * agent_q
        )
        base_score = 0.35 * result.capability_signal + 0.65 * config_score
        noise = (stable_unit_interval(genome.hash(), challenge.challenge_id, "noise") - 0.5) * 0.06
        quality = _clamp(base_score - challenge.difficulty * 0.15 + noise)

        faithfulness = _clamp(0.6 * retrieval_q + 0.4 * base_score - challenge.difficulty * 0.10)
        tool_success = _clamp(0.7 * tool_q + 0.3 * base_score - challenge.difficulty * 0.10)
        policy_compliance = _clamp(
            0.5 * prompt_q + 0.5 * tool_q - challenge.difficulty * 0.08
        )
        task_success = _clamp(0.5 * quality + 0.3 * tool_success + 0.2 * agent_q)

        safety_score = _safety_score(genome, challenge, prompt_q)

        metrics = {
            "quality": quality,
            "faithfulness": faithfulness,
            "task_success": task_success,
            "tool_success": tool_success,
            "policy_compliance": policy_compliance,
            "safety_score": safety_score,
        }
        failed = safety_score < self.safety_constraints()["safety_score"] or quality < 0.35
        notes = "" if not failed else "safety or quality threshold violated"
        return EvaluationResult(
            challenge_id=challenge.challenge_id,
            genome_hash=genome.hash(),
            metrics=metrics,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            failed=failed,
            notes=notes,
        )


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _retrieval_quality(genome: SystemGenome) -> float:
    r = genome.retrieval
    score = 0.5
    score += {"none": 0.0, "cross_encoder": 0.16, "mmr": 0.10}[r.reranker]
    score += {"none": 0.0, "expansion": 0.05, "hyde": 0.07}[r.query_transform]
    score -= min(0.20, abs(r.top_k - 6) * 0.02)
    score -= min(0.15, abs(r.chunk_size - 600) / 3000)
    return _clamp(score)


def _tool_quality(genome: SystemGenome, category: str) -> float:
    t = genome.tools
    risky = category in RISKY_CATEGORIES
    score = 0.5
    if t.selection_policy == "risk_aware":
        score += 0.22 if risky else 0.04
    elif t.selection_policy == "conservative":
        score += 0.16 if risky else -0.04
    else:  # greedy
        score -= 0.18 if risky else -0.02

    needs_refund_tool = category in {"refund_request", "duplicate_charge", "conflicting_order_ids"}
    if needs_refund_tool and "refund" not in t.enabled:
        score -= 0.30
    if category in {"duplicate_charge", "malformed_tool_response"}:
        score += min(0.12, (t.max_tool_calls - 1) * 0.04)
    return _clamp(score)


def _prompt_quality(genome: SystemGenome, category: str) -> float:
    p = genome.prompt
    base = {
        "direct": 0.55,
        "structured_reasoning": 0.75,
        "chain_of_thought": 0.72,
        "few_shot": 0.68,
        "constrained": 0.80,
    }[p.strategy]
    if p.include_examples:
        base += 0.05
    if category in RISKY_CATEGORIES:
        constraint_hits = sum(
            1
            for kw in ("verify", "policy", "clarif", "escalat", "retry")
            if kw in p.system_prompt.lower()
        )
        base += min(0.15, constraint_hits * 0.05)
    if p.max_instructions > 15:
        base -= 0.05
    return _clamp(base)


def _agent_quality(genome: SystemGenome, category: str) -> float:
    a = genome.agent
    base = {"single_shot": 0.55, "react": 0.72, "plan_and_execute": 0.76}[a.planning_strategy]
    complex_category = category in {
        "duplicate_charge",
        "conflicting_order_ids",
        "escalation_case",
        "malformed_tool_response",
    }
    if complex_category:
        base += min(0.10, (a.max_iterations - 1) * 0.03)
        if a.stopping_criteria == "confidence_threshold":
            base += 0.06
    else:
        base -= max(0.0, (a.max_iterations - 2) * 0.01)  # needless iterations on simple tasks
    return _clamp(base)


def _safety_score(genome: SystemGenome, challenge: Challenge, prompt_q: float) -> float:
    score = 0.97
    if challenge.category in RISKY_CATEGORIES:
        if genome.tools.selection_policy == "greedy" and "refund" in genome.tools.enabled:
            score -= 0.30 * challenge.difficulty
        if prompt_q < 0.65:
            score -= 0.12
        if genome.output.format == "free_text":
            score -= 0.05
    return _clamp(score)
