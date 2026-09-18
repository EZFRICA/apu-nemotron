"""Escalation records. All immutable: "resolved" is a join at read time, never a mutated flag."""

from dataclasses import dataclass
from datetime import datetime

from apu.guardrails.policy import parse_class_id


@dataclass(frozen=True)
class EscalationEvent:
    event_id: str
    student_id: str
    class_id: str  # "establishment_id:class_code"
    session_id: str
    attempt_number_in_session: int
    off_topic_request_text: str
    triggered_at: datetime

    def __post_init__(self) -> None:
        parse_class_id(self.class_id)

    @property
    def establishment_id(self) -> str:
        return self.class_id.split(":", 1)[0]


@dataclass(frozen=True)
class EscalationResolution:
    event_id: str
    resolved_by: str
    resolved_at: datetime
    note: str | None = None


@dataclass(frozen=True)
class EscalationCluster:
    cluster_id: int
    event_ids: list[str]
    # Text of the cluster's earliest event. TODO: a short summary by Nemotron Nano
    # (EXTRACTION_MODEL); deliberately not built yet.
    representative_text: str

    @property
    def size(self) -> int:
        return len(self.event_ids)


@dataclass(frozen=True)
class EscalationClusterSnapshot:
    class_id: str
    computed_at: datetime
    clusters: list[EscalationCluster]
