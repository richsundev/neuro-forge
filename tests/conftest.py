import pytest

from neuroforge.datasets.evolution import seed_dataset
from neuroforge.domains import get_domain
from neuroforge.genomes.schema import SystemGenome


@pytest.fixture
def forge_support_domain():
    return get_domain("forge-support")


@pytest.fixture
def baseline_genome():
    return SystemGenome(system_id="support-agent", version=1)


@pytest.fixture
def small_dataset(forge_support_domain):
    return seed_dataset(forge_support_domain, "support-test", n=40, seed=1)
