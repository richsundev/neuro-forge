"""Domain events (section 33). Persisted as an append-only JSON-lines log per experiment so the
event history survives a crash and can be replayed for the dashboard/API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_TYPES = (
    "ExperimentCreated",
    "CandidateGenerated",
    "CandidateEvaluated",
    "GenerationCompleted",
    "BenchmarkUpdated",
    "PromotionRequested",
    "CandidatePromoted",
    "CandidateRejected",
    "CanaryStarted",
    "RollbackTriggered",
    "ExperimentStopped",
)


@dataclass
class Event:
    event_type: str
    experiment_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type '{self.event_type}'")


class EventLog:
    """Append-only event log. Backed by a JSONL file so it's recoverable across process restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[Event] = []
        self._tail_checked = False
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    self._events.append(Event(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    # A process killed mid-append leaves a torn final line. The log is append-only
                    # and every complete event before it is still valid, so skip it rather than
                    # making the whole experiment unresumable (see scripts/failures/worker_crash.py).
                    continue

    def _terminate_torn_tail(self) -> None:
        """Make sure the file ends in a newline before the first append, so a new event isn't glued
        onto a torn fragment (which would corrupt it too). Done at the first *write*, not when the
        log is opened: opening happens before the experiment's run lock is taken, and another live
        writer's in-progress line must not be "repaired" out from under it."""
        if self._tail_checked:
            return
        self._tail_checked = True
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                last = f.read(1)
            if last != b"\n":
                with self.path.open("a") as f:
                    f.write("\n")

    def emit(self, event: Event) -> None:
        self._terminate_torn_tail()
        self._events.append(event)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def all(self) -> list[Event]:
        return list(self._events)

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.event_type == event_type]
