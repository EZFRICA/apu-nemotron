"""The assignment registry: who teaches which class, who administers which establishment.

This registry is the ONLY source of a requester's role and scope. A role or class declared
by the caller (a header, a body field, a query parameter) is never trusted. It is loaded
once at startup, like ClassPolicy, from config.TEACHER_ASSIGNMENTS_PATH.
"""

import json
import threading
from dataclasses import dataclass
from typing import Literal

from apu import config
from apu.guardrails.policy import parse_class_id

Role = Literal["teacher", "establishment_admin"]
_ROLES = ("teacher", "establishment_admin")


@dataclass(frozen=True)
class TeacherAssignment:
    requester_id: str
    role: Role
    establishment_id: str
    class_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"Unknown role {self.role!r} for {self.requester_id!r}.")
        if self.role == "teacher":
            if not self.class_id:
                raise ValueError(f"Teacher {self.requester_id!r} has no class_id.")
            establishment_id, _ = parse_class_id(self.class_id)
            if establishment_id != self.establishment_id:
                raise ValueError(
                    f"Teacher {self.requester_id!r}: class {self.class_id!r} is not in "
                    f"establishment {self.establishment_id!r}."
                )
        elif self.class_id is not None:
            raise ValueError(
                f"Establishment admin {self.requester_id!r} must not be tied to a class."
            )


class AssignmentRegistry:
    def __init__(self, assignments: list[TeacherAssignment]) -> None:
        by_requester: dict[str, TeacherAssignment] = {}
        for assignment in assignments:
            if assignment.requester_id in by_requester:
                # One role per requester: two entries would make the effective scope depend
                # on file order.
                raise ValueError(f"Duplicate assignment for {assignment.requester_id!r}.")
            by_requester[assignment.requester_id] = assignment
        self._by_requester = by_requester

    @classmethod
    def load(cls, path: str) -> "AssignmentRegistry":
        with open(path, encoding="utf-8") as registry_file:
            entries = json.load(registry_file)
        return cls([TeacherAssignment(**entry) for entry in entries])

    def lookup(self, requester_id: str) -> TeacherAssignment | None:
        return self._by_requester.get(requester_id)

    def classes_in_establishment(self, establishment_id: str) -> list[str]:
        return sorted({
            assignment.class_id
            for assignment in self._by_requester.values()
            if assignment.role == "teacher" and assignment.establishment_id == establishment_id
        })


_registry: AssignmentRegistry | None = None
_registry_lock = threading.Lock()


def load_assignment_registry(path: str | None = None) -> AssignmentRegistry:
    global _registry
    with _registry_lock:
        _registry = AssignmentRegistry.load(path or config.TEACHER_ASSIGNMENTS_PATH)
        return _registry


def set_assignment_registry(registry: AssignmentRegistry | None) -> None:
    global _registry
    with _registry_lock:
        _registry = registry


def get_assignment_registry() -> AssignmentRegistry:
    if _registry is None:
        return load_assignment_registry()
    return _registry


def lookup_assignment(requester_id: str) -> TeacherAssignment | None:
    return get_assignment_registry().lookup(requester_id)
