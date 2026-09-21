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
        if self.path.exists():
            raw = self.path.read_text()
            if raw and not raw.endswith("\n"):
                # Terminate a torn final line so the next append starts on a fresh line instead of
                # being glued onto the fragment (which would corrupt that event too).
                with self.path.open("a") as f:
                    f.write("\n")
                raw += "\n"
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    self._events.append(Event(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    # A process killed mid-append leaves a torn final line. The log is append-only
                    # and every complete event before it is still valid, so skip it rather than
                    # making the whole experiment unresumable (see scripts/failures/worker_crash.py).
                    continue

    def emit(self, event: Event) -> None:
        self._events.append(event)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def all(self) -> list[Event]:
        return list(self._events)

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.event_type == event_type]
