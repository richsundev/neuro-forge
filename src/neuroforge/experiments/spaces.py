"""Default search spaces per domain, expressed directly in genome field paths so every point a
strategy proposes maps 1:1 onto a `SystemGenome.derive()` override — see engine.py."""

from __future__ import annotations

from neuroforge.optimization.search_space import ParamSpec, SearchSpace

BASE_SUPPORT_PROMPT = "You are a helpful customer support assistant."

SUPPORT_PROMPT_VARIANTS = [
    BASE_SUPPORT_PROMPT,
    BASE_SUPPORT_PROMPT
    + "\n- Never promise a refund without verifying the order ID against the order lookup tool.",
    BASE_SUPPORT_PROMPT
    + "\n- Never promise a refund without verifying the order ID against the order lookup tool."
    + "\n- If policy and the customer's claim conflict, defer to policy and explain why."
    + "\n- When information is missing or conflicting, ask a clarifying question instead of guessing.",
    BASE_SUPPORT_PROMPT
    + "\n- Never promise a refund without verifying the order ID against the order lookup tool."
    + "\n- If policy and the customer's claim conflict, defer to policy and explain why."
    + "\n- When information is missing or conflicting, ask a clarifying question instead of guessing."
    + "\n- Escalate repeated or high-risk complaints to a human agent rather than resolving unilaterally."
    + "\n- Always state the specific policy clause used to justify a decision."
    + "\n- If a tool call fails or times out, retry once before telling the customer.",
]


def forge_support_search_space() -> SearchSpace:
    return SearchSpace(
        parameters={
            "model.name": ParamSpec(
                type="categorical", values=["fast-cheap", "reasoning-small", "reasoning-large"]
            ),
            "model.temperature": ParamSpec(type="float", min=0.0, max=0.8),
            "model.reasoning_effort": ParamSpec(type="categorical", values=["low", "medium", "high"]),
            "prompt.system_prompt": ParamSpec(type="categorical", values=SUPPORT_PROMPT_VARIANTS),
            "prompt.strategy": ParamSpec(
                type="categorical",
                values=["direct", "structured_reasoning", "chain_of_thought", "few_shot", "constrained"],
            ),
            "prompt.include_examples": ParamSpec(type="categorical", values=[True, False]),
            "retrieval.top_k": ParamSpec(type="integer", min=3, max=10),
            "retrieval.chunk_size": ParamSpec(type="categorical", values=[400, 600, 800, 1000]),
            "retrieval.reranker": ParamSpec(type="categorical", values=["none", "cross_encoder", "mmr"]),
            "tools.enabled": ParamSpec(
                type="categorical",
                values=[
                    ["search"],
                    ["search", "refund"],
                    ["search", "refund", "ticket"],
                ],
            ),
            "tools.selection_policy": ParamSpec(
                type="categorical", values=["greedy", "risk_aware", "conservative"]
            ),
            "agent.planning_strategy": ParamSpec(
                type="categorical", values=["single_shot", "react", "plan_and_execute"]
            ),
            "agent.max_iterations": ParamSpec(type="integer", min=1, max=5),
            "agent.stopping_criteria": ParamSpec(
                type="categorical",
                values=["first_answer", "confidence_threshold", "max_iterations"],
            ),
        }
    )


def default_search_space() -> SearchSpace:
    """A smaller, domain-agnostic space for domains without a hand-tuned one (sql-agent,
    research-agent) — still expressed as genome field paths per engine.py's design."""
    return SearchSpace(
        parameters={
            "model.name": ParamSpec(
                type="categorical", values=["fast-cheap", "reasoning-small", "reasoning-large"]
            ),
            "model.temperature": ParamSpec(type="float", min=0.0, max=0.8),
            "prompt.strategy": ParamSpec(
                type="categorical",
                values=["direct", "structured_reasoning", "chain_of_thought", "few_shot", "constrained"],
            ),
            "retrieval.top_k": ParamSpec(type="integer", min=3, max=10),
            "retrieval.reranker": ParamSpec(type="categorical", values=["none", "cross_encoder", "mmr"]),
            "output.format": ParamSpec(type="categorical", values=["free_text", "json_schema", "markdown"]),
        }
    )


def search_space_for_domain(domain_name: str) -> SearchSpace:
    if domain_name == "forge-support":
        return forge_support_search_space()
    return default_search_space()
