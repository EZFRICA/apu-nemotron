"""
Source citations appended to an answer.

Target: apu/modality/citations.py
"""

from apu.modality.citations import Source, render_answer, written_source_list

SOURCES = [
    Source("Photosynthesis — Wikipedia", "https://en.wikipedia.org/wiki/Photosynthesis"),
    Source("Photosynthesis | Lumni", "https://www.lumni.fr/video/la-photosynthese"),
    Source("Chlorophyll — Wikipedia", "https://en.wikipedia.org/wiki/Chlorophyll"),
]


def test_an_answer_ends_with_its_numbered_sources():
    rendered = render_answer("Photosynthesis produces glucose.", SOURCES)
    assert rendered.startswith("Photosynthesis produces glucose.")
    assert rendered.endswith(
        "Sources:\n"
        "1. Photosynthesis — Wikipedia — https://en.wikipedia.org/wiki/Photosynthesis\n"
        "2. Photosynthesis | Lumni — https://www.lumni.fr/video/la-photosynthese\n"
        "3. Chlorophyll — Wikipedia — https://en.wikipedia.org/wiki/Chlorophyll"
    )


def test_without_sources_the_answer_is_unchanged():
    assert render_answer("Answer.", []) == "Answer."
    assert written_source_list([]) == "Sources:"


def test_a_source_is_named_after_its_site():
    assert Source("x", "https://fr.wikipedia.org/wiki/Fraction").site_name == "Wikipedia"
    assert Source("x", "https://www.education.gouv.fr/page").site_name == "the French Ministry of Education"
    assert Source("x", "https://www.example-school.org/page").site_name == "Example-school"
