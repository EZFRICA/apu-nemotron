"""
Interaction modes and source citations per output channel.

Target: apu/modality/mode.py, apu/modality/citations.py
"""

import pytest

from apu.modality.citations import Source, render_answer, spoken_attribution
from apu.modality.mode import SUPPORTED_COMBINATIONS, InputChannel, InteractionMode, OutputChannel

SOURCES = [
    Source("Photosynthèse — Wikipédia", "https://fr.wikipedia.org/wiki/Photosynth%C3%A8se"),
    Source("La photosynthèse | Lumni", "https://www.lumni.fr/video/la-photosynthese"),
    Source("Chlorophylle — Wikipédia", "https://fr.wikipedia.org/wiki/Chlorophylle"),
]


# ── modes ────────────────────────────────────────────────────────────────────

def test_exactly_five_combinations_are_supported():
    assert len(SUPPORTED_COMBINATIONS) == 5


@pytest.mark.parametrize("input_channel,output_channel", sorted(SUPPORTED_COMBINATIONS))
def test_every_supported_combination_builds(input_channel, output_channel):
    mode = InteractionMode(input_channel, output_channel)
    assert (mode.input_channel, mode.output_channel) == (input_channel, output_channel)


@pytest.mark.parametrize("input_channel,output_channel", [
    ("braille", "voice"), ("braille", "text"), ("text", "braille"), ("voice", "braille"),
])
def test_unsupported_combinations_are_refused(input_channel, output_channel):
    with pytest.raises(ValueError, match="Unsupported interaction mode"):
        InteractionMode(input_channel, output_channel)


def test_a_text_channel_implies_a_display():
    assert InteractionMode("voice", "text").text_display_available is True
    assert InteractionMode("text", "voice").text_display_available is True
    assert InteractionMode("voice", "voice").text_display_available is False
    assert InteractionMode("voice", "voice", text_display_available=True).text_display_available is True


def test_channels_accept_plain_strings():
    mode = InteractionMode("braille", "braille")
    assert mode.input_channel is InputChannel.BRAILLE
    assert mode.output_channel is OutputChannel.BRAILLE
    assert mode.is_braille and not mode.is_spoken


# ── citations ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", [InteractionMode("text", "text"), InteractionMode("braille", "braille")])
def test_text_and_braille_get_a_source_list_at_the_end(mode):
    rendered = render_answer("La photosynthèse produit du glucose.", SOURCES, mode)
    assert rendered.spoken is None
    assert rendered.written.startswith("La photosynthèse produit du glucose.")
    assert rendered.written.endswith(
        "Sources :\n"
        "1. Photosynthèse — Wikipédia — https://fr.wikipedia.org/wiki/Photosynth%C3%A8se\n"
        "2. La photosynthèse | Lumni — https://www.lumni.fr/video/la-photosynthese\n"
        "3. Chlorophylle — Wikipédia — https://fr.wikipedia.org/wiki/Chlorophylle"
    )


def test_voice_names_sources_aloud_without_urls():
    rendered = render_answer("La photosynthèse produit du glucose.", SOURCES, InteractionMode("voice", "voice"))
    assert rendered.spoken == "D'après Wikipédia et Lumni : La photosynthèse produit du glucose."
    assert "http" not in rendered.spoken
    assert rendered.written is None, "no text channel in this session"


def test_voice_with_a_text_display_also_gets_the_written_list():
    rendered = render_answer(
        "La photosynthèse produit du glucose.", SOURCES,
        InteractionMode("voice", "voice", text_display_available=True),
    )
    assert "http" not in rendered.spoken
    assert "Sources :" in rendered.written and "https://www.lumni.fr" in rendered.written


def test_text_input_with_voice_output_gets_both():
    rendered = render_answer("Réponse.", SOURCES, InteractionMode("text", "voice"))
    assert rendered.spoken.startswith("D'après")
    assert "Sources :" in rendered.written


def test_without_sources_the_answer_is_unchanged():
    assert render_answer("Réponse.", [], InteractionMode("text", "text")).written == "Réponse."
    assert render_answer("Réponse.", [], InteractionMode("voice", "voice")).spoken == "Réponse."


def test_unknown_sites_are_named_after_their_domain():
    assert spoken_attribution([Source("x", "https://www.example-ecole.org/page")]) == "D'après Example-ecole"
    assert spoken_attribution([
        Source("a", "https://a.fr"), Source("b", "https://b.fr"), Source("c", "https://c.fr"),
    ]) == "D'après A, B et C"
