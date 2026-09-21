"""In-memory genome store with lineage graph queries. Backed by the DB layer in production."""

from __future__ import annotations

from dataclasses import dataclass, field

from neuroforge.genomes.schema import MutationRecord, SystemGenome


def _clip(value: object, limit: int = 48) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe(m: MutationRecord) -> str:
    """`mutation_type` alone is useless in the graph — every search-proposed change has the same
    type ("evolutionary.propose"), so a genome with 14 changes listed the same string 14 times."""
    return f"{m.field_path}: {_clip(m.old_value)} → {_clip(m.new_value)}"


@dataclass
class GenomeStore:
    """Keeps every genome ever created for an application, indexed for lineage traversal."""

    _by_hash: dict[str, SystemGenome] = field(default_factory=dict)
    _children: dict[str, list[str]] = field(default_factory=dict)
    _by_system: dict[str, list[str]] = field(default_factory=dict)

    def add(self, genome: SystemGenome) -> SystemGenome:
        h = genome.hash()
        if h in self._by_hash:
            return self._by_hash[h]
        self._by_hash[h] = genome
        self._by_system.setdefault(genome.system_id, []).append(h)
        if genome.parent_hash:
            self._children.setdefault(genome.parent_hash, []).append(h)
        return genome

    def get(self, genome_hash: str) -> SystemGenome | None:
        return self._by_hash.get(genome_hash)

    def children_of(self, genome_hash: str) -> list[SystemGenome]:
        return [self._by_hash[h] for h in self._children.get(genome_hash, [])]

    def all_for_system(self, system_id: str) -> list[SystemGenome]:
        return [self._by_hash[h] for h in self._by_system.get(system_id, [])]

    def ancestry(self, genome_hash: str) -> list[SystemGenome]:
        """Walk from a genome back to its root, oldest first."""
        chain: list[SystemGenome] = []
        cursor = self._by_hash.get(genome_hash)
        while cursor is not None:
            chain.append(cursor)
            cursor = self._by_hash.get(cursor.parent_hash) if cursor.parent_hash else None
        return list(reversed(chain))

    def roots(self, system_id: str) -> list[SystemGenome]:
        return [g for g in self.all_for_system(system_id) if g.parent_hash is None]

    def lineage_graph(self, system_id: str) -> dict[str, object]:
        """Serializable node/edge graph for the Evolution Graph UI."""
        genomes = self.all_for_system(system_id)
        nodes = [
            {
                "hash": g.hash(),
                "version": g.version,
                "generation": g.generation,
                "status": g.status.value,
                "parent_hash": g.parent_hash,
                "mutations": [_describe(m) for m in g.mutations],
            }
            for g in genomes
        ]
        edges = [
            {"from": g.parent_hash, "to": g.hash()} for g in genomes if g.parent_hash is not None
        ]
        return {"nodes": nodes, "edges": edges}


_GLOBAL_STORE = GenomeStore()


def get_default_store() -> GenomeStore:
    return _GLOBAL_STORE
