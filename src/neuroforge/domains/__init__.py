from neuroforge.domains.base import ApplicationDomain, Challenge, EvaluationResult
from neuroforge.domains.forge_support import ForgeSupportDomain
from neuroforge.domains.research_agent import ResearchAgentDomain
from neuroforge.domains.sql_agent import SQLAgentDomain

DOMAIN_REGISTRY: dict[str, type[ApplicationDomain]] = {
    "forge-support": ForgeSupportDomain,
    "sql-agent": SQLAgentDomain,
    "research-agent": ResearchAgentDomain,
}


def get_domain(name: str) -> ApplicationDomain:
    cls = DOMAIN_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown domain '{name}'. Known: {list(DOMAIN_REGISTRY)}")
    return cls()


__all__ = [
    "DOMAIN_REGISTRY",
    "ApplicationDomain",
    "Challenge",
    "EvaluationResult",
    "ForgeSupportDomain",
    "ResearchAgentDomain",
    "SQLAgentDomain",
    "get_domain",
]
