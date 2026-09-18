"""The system prompts the cloud registry ships, validated before they become instructions.

`prompts.json` is written by the registry sync and holds the tutor's persona (`system_tutor`)
and per-class guidelines. That makes the registry a source of INSTRUCTIONS, not only of
content: whoever can write to the bucket writes what the tutor is. The device cannot verify
intent, but it can refuse what is obviously not a prompt file — a wrong shape, a value that
is not text, or one long enough to push everything else out of the model's context — and
fall back to the built-in persona instead of loading it blindly.

Read at every turn, deliberately: a refreshed file applies to the next question, and a file
that was tampered with locally is caught on the same path as one that arrived broken.
"""

import json
import os

from apu import config
from apu.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TUTOR_PERSONA = "You are Akili, an expert academic tutor."
TUTOR_PERSONA_KEY = "system_tutor"


def prompts_path() -> str:
    """Where the registry sync writes the prompts, next to the L3 store."""
    return os.path.join(os.path.dirname(config.LANCE_DB_PATH), "prompts.json")


def _validated_text(key: str, value) -> str | None:
    if not isinstance(value, str):
        logger.warning("Registry prompt %r ignored: %s, not text.", key, type(value).__name__)
        return None
    if len(value) > config.REGISTRY_PROMPT_MAX_CHARS:
        logger.warning(
            "Registry prompt %r ignored: %d characters, over the %d limit.",
            key, len(value), config.REGISTRY_PROMPT_MAX_CHARS,
        )
        return None
    return value


def load_registry_prompts(class_level: str) -> tuple[str, str]:
    """
    (tutor persona, class guidelines) from the registry, falling back to the built-in persona.

    Anything the file does not provide, or provides in a shape this device refuses, simply
    does not apply; a broken prompts file degrades to the default tutor rather than to no
    tutor at all.
    """
    path = prompts_path()
    if not os.path.exists(path):
        return DEFAULT_TUTOR_PERSONA, ""
    try:
        with open(path, encoding="utf-8") as prompts_file:
            data = json.load(prompts_file)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Registry prompts ignored (%s): %s", path, error)
        return DEFAULT_TUTOR_PERSONA, ""

    if not isinstance(data, dict):
        logger.warning("Registry prompts ignored: the file is not a JSON object.")
        return DEFAULT_TUTOR_PERSONA, ""

    persona = _validated_text(TUTOR_PERSONA_KEY, data.get(TUTOR_PERSONA_KEY, DEFAULT_TUTOR_PERSONA))
    guidelines = _validated_text(class_level, data.get(class_level, ""))
    return persona or DEFAULT_TUTOR_PERSONA, guidelines or ""
