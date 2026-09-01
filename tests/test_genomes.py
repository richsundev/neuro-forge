import pytest
from pydantic import ValidationError

from neuroforge.genomes.lineage import GenomeStore
from neuroforge.genomes.schema import MutationRecord, PromotionStatus, SystemGenome


def test_genome_hash_is_deterministic():
    a = SystemGenome(system_id="s", version=1)
    b = SystemGenome(system_id="s", version=1)
    assert a.hash() == b.hash()


def test_genome_hash_changes_with_content():
    a = SystemGenome(system_id="s", version=1)
    b = a.derive(mutations=[], overrides={"model.temperature": 0.9}, new_version=2)
    assert a.hash() != b.hash()


def test_genome_hash_ignores_metadata_fields():
    """Version/status/created_at differences shouldn't change the content hash."""
    a = SystemGenome(system_id="s", version=1)
    b = SystemGenome(system_id="s", version=2, status=PromotionStatus.PROMOTED)
    assert a.hash() == b.hash()


def test_derive_sets_lineage_fields():
    parent = SystemGenome(system_id="s", version=1)
    child = parent.derive(
        mutations=[
            MutationRecord(
                mutation_type="model.change_temperature",
                field_path="model.temperature",
                old_value=0.2,
                new_value=0.5,
                reason="test",
            )
        ],
        overrides={"model.temperature": 0.5},
        new_version=2,
    )
    assert child.parent_hash == parent.hash()
    assert child.generation == parent.generation + 1
    assert child.model.temperature == 0.5
    assert child.status == PromotionStatus.GENERATED
    assert len(child.mutations) == 1


def test_genome_immutable():
    g = SystemGenome(system_id="s", version=1)
    with pytest.raises(ValidationError):
        g.version = 99  # type: ignore[misc]


def test_lineage_store_ancestry_and_children():
    store = GenomeStore()
    root = store.add(SystemGenome(system_id="s", version=1))
    child = store.add(root.derive(mutations=[], overrides={"model.temperature": 0.4}, new_version=2))
    grandchild = store.add(child.derive(mutations=[], overrides={"retrieval.top_k": 8}, new_version=3))

    ancestry = store.ancestry(grandchild.hash())
    assert [g.hash() for g in ancestry] == [root.hash(), child.hash(), grandchild.hash()]
    assert store.children_of(root.hash())[0].hash() == child.hash()
    assert store.roots("s") == [root]


def test_lineage_graph_serializable():
    store = GenomeStore()
    root = store.add(SystemGenome(system_id="s2", version=1))
    store.add(root.derive(mutations=[], overrides={"model.temperature": 0.4}, new_version=2))
    graph = store.lineage_graph("s2")
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
