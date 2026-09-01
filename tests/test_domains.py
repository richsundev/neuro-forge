from neuroforge.genomes.schema import SystemGenome
from neuroforge.providers.mock import MockLLMProvider


def test_evaluate_is_deterministic(forge_support_domain, baseline_genome):
    provider = MockLLMProvider()
    challenge = forge_support_domain.generate_cases(1, (0.3, 0.3), seed=9)[0]
    r1 = forge_support_domain.evaluate(baseline_genome, challenge, provider)
    r2 = forge_support_domain.evaluate(baseline_genome, challenge, provider)
    assert r1.metrics == r2.metrics
    assert r1.cost_usd == r2.cost_usd
    assert r1.latency_ms == r2.latency_ms


def test_better_config_scores_higher_on_risky_category(forge_support_domain):
    provider = MockLLMProvider()
    weak = SystemGenome(system_id="s", version=1)
    strong = weak.derive(
        mutations=[],
        overrides={
            "prompt.strategy": "constrained",
            "prompt.system_prompt": (
                "You are a support assistant.\n- verify order id\n- policy first\n"
                "- ask a clarifying question if information conflicts\n- escalate if needed\n- retry once"
            ),
            "tools.enabled": ["search", "refund"],
            "tools.selection_policy": "risk_aware",
        },
        new_version=2,
    )
    challenges = [c for c in forge_support_domain.generate_cases(20, (0.3, 0.5), seed=4) if c.category == "refund_request"]
    assert challenges

    weak_scores = [forge_support_domain.evaluate(weak, c, provider).metrics["quality"] for c in challenges]
    strong_scores = [forge_support_domain.evaluate(strong, c, provider).metrics["quality"] for c in challenges]
    assert sum(strong_scores) / len(strong_scores) > sum(weak_scores) / len(weak_scores)


def test_safety_penalizes_greedy_refund_policy(forge_support_domain):
    provider = MockLLMProvider()
    genome = SystemGenome(system_id="s", version=1).derive(
        mutations=[],
        overrides={"tools.enabled": ["search", "refund"], "tools.selection_policy": "greedy"},
        new_version=2,
    )
    challenges = [
        c for c in forge_support_domain.generate_cases(10, (0.7, 0.9), seed=6) if c.category == "duplicate_charge"
    ]
    results = [forge_support_domain.evaluate(genome, c, provider) for c in challenges]
    assert any(r.metrics["safety_score"] < 0.85 for r in results)


def test_validate_rejects_malformed_challenge(forge_support_domain):
    challenges = forge_support_domain.generate_cases(1, (0.1, 0.1), seed=1)
    bad = challenges[0].model_copy(update={"input": "   "})
    assert not forge_support_domain.validate(bad)
