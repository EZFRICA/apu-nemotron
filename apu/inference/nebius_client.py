"""Thin wrapper around the Nebius Token Factory OpenAI-compatible API.

Two entry points are exposed deliberately, rather than one generic "call the model"
function, because the APU pipeline always uses two different Nemotron sizes for two
different jobs: a larger model for the answer the user actually reads, and a smaller
one for the background memory-extraction call that never reaches the user.
"""

import os

from openai import OpenAI

from apu.config import EXTRACTION_MODEL, MAIN_MODEL, NEBIUS_BASE_URL


def _build_client() -> OpenAI:
    # Fails loudly on a missing key instead of silently falling back to no auth,
    # since a misconfigured key here means every call in the pipeline breaks.
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Get one at https://tokenfactory.nebius.com/ "
            "and add it to your .env file."
        )
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)


# Module-level client, reused across calls rather than rebuilt per request.
_client = _build_client()


def call_main_model(messages: list[dict], **kwargs) -> str:
    """Answer call the user sees. Defaults to the larger Nemotron for reasoning quality."""
    response = _client.chat.completions.create(
        model=kwargs.pop("model", MAIN_MODEL),
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content


def call_main_model_message(messages: list[dict], **kwargs):
    """Answer call returning the whole assistant message, not only its text.

    Needed for native tool calling (Method A, see HACKATHON.md): when Nemotron asks for a
    tool, `content` is None and the request is in `tool_calls`, which call_main_model
    would drop. The reasoning trace, when present, is in `model_extra["reasoning_content"]`.
    """
    response = _client.chat.completions.create(
        model=kwargs.pop("model", MAIN_MODEL),
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message


def call_extraction_model(messages: list[dict], **kwargs) -> str:
    """Background memory-extraction call, off the critical path.

    Defaults to Nemotron Nano: this call historically dominated turn latency and
    variance in the pre-Nemotron benchmarks, so keeping it on the cheapest capable
    model matters more here than for the main answer call.
    """
    response = _client.chat.completions.create(
        model=kwargs.pop("model", EXTRACTION_MODEL),
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content
