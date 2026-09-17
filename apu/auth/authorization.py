"""Authorization of teacher and admin views. Scope always comes from the assignment registry."""

from dataclasses import dataclass
from typing import Literal

from apu.auth import assignments


@dataclass(frozen=True)
class TeacherViewScope:
    requester_id: str
    role: Literal["teacher", "establishment_admin"]
    establishment_id: str
    class_id: str | None = None  # required when role == "teacher"


def authorize_view(requester_id: str, target_class_id: str) -> TeacherViewScope:
    assignment = assignments.lookup_assignment(requester_id)
    if assignment is None:
        raise PermissionError(f"{requester_id} n'a aucun rôle enregistré.")
    if assignment.role == "establishment_admin":
        if not target_class_id.startswith(f"{assignment.establishment_id}:"):
            raise PermissionError("Classe hors de l'établissement de l'admin.")
    elif assignment.role == "teacher" and assignment.class_id != target_class_id:
        raise PermissionError("Classe hors du périmètre du prof.")
    return TeacherViewScope(
        requester_id, assignment.role, assignment.establishment_id, assignment.class_id
    )
