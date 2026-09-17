"""Per-class policy: the only thing that varies between classes in the guardrails.

The topical rail itself is one shared Colang config. What differs per class is data, loaded
into the session when it opens: extra excluded domains for web search, and the off-topic
escalation threshold. Nothing is compiled per class.
"""

import json
import threading
from dataclasses import dataclass
from datetime import datetime

from apu import config


def parse_class_id(class_id: str) -> tuple[str, str]:
    """'etablissement_id:classe_code' -> (etablissement_id, classe_code)."""
    establishment_id, separator, class_code = class_id.partition(":")
    if not separator or not establishment_id or not class_code:
        raise ValueError(
            f"Invalid class_id {class_id!r}: expected 'etablissement_id:classe_code', "
            "e.g. 'lycee-cocody:3eA'."
        )
    return establishment_id, class_code


@dataclass(frozen=True)
class ClassPolicy:
    class_id: str  # namespaced "etablissement_id:classe_code", e.g. "lycee-cocody:3eA"
    teacher_id: str
    tavily_excluded_domains: list[str]  # added to GLOBAL_EXCLUDED_DOMAINS, never replaces it
    escalation_threshold: int
    updated_at: datetime

    def __post_init__(self) -> None:
        parse_class_id(self.class_id)
        if self.escalation_threshold < 1:
            raise ValueError(f"{self.class_id}: escalation_threshold must be at least 1.")
        # A private copy: the dataclass is frozen, but a list handed in by the caller could
        # still be mutated from outside after the session captured it.
        object.__setattr__(self, "tavily_excluded_domains", list(self.tavily_excluded_domains))


class ClassPolicyRegistry:
    def __init__(self, policies: list[ClassPolicy]) -> None:
        by_class: dict[str, ClassPolicy] = {}
        for policy in policies:
            if policy.class_id in by_class:
                raise ValueError(f"Duplicate class policy for {policy.class_id!r}.")
            by_class[policy.class_id] = policy
        self._by_class = by_class

    @classmethod
    def load(cls, path: str) -> "ClassPolicyRegistry":
        with open(path, encoding="utf-8") as registry_file:
            entries = json.load(registry_file)
        return cls([
            ClassPolicy(**{**entry, "updated_at": datetime.fromisoformat(entry["updated_at"])})
            for entry in entries
        ])

    def get(self, class_id: str) -> ClassPolicy:
        try:
            return self._by_class[class_id]
        except KeyError:
            raise KeyError(f"No class policy for {class_id!r}.") from None


_registry: ClassPolicyRegistry | None = None
_registry_lock = threading.Lock()


def load_class_policy_registry(path: str | None = None) -> ClassPolicyRegistry:
    global _registry
    with _registry_lock:
        _registry = ClassPolicyRegistry.load(path or config.CLASS_POLICIES_PATH)
        return _registry


def set_class_policy_registry(registry: ClassPolicyRegistry | None) -> None:
    global _registry
    with _registry_lock:
        _registry = registry


def get_class_policy_registry() -> ClassPolicyRegistry:
    if _registry is None:
        return load_class_policy_registry()
    return _registry
