"""
Braille translation (liblouis) and the embosser simulator.

Target: apu/modality/braille/translator.py, apu/modality/braille/embosser_simulator.py

Translation tests skip when the liblouis C library is not installed; the Braille ASCII
mapping and the embosser layout do not need it.
"""

import pytest

from apu.modality.braille import _liblouis
from apu.modality.braille.embosser_simulator import (
    EmbosserError,
    NetworkEmbosser,
    SimulatedEmbosser,
)
from apu.modality.braille.translator import (
    BRAILLE_ASCII,
    BrailleEncoding,
    BrailleEncodingError,
    BrailleGrade,
    BrailleTranslator,
    unicode_to_braille_ascii,
)


@pytest.fixture
def liblouis():
    try:
        _liblouis.load_library()
    except _liblouis.LiblouisUnavailable as error:
        pytest.skip(str(error))
    return _liblouis


def _is_unicode_braille(text: str) -> bool:
    return all(0x2800 <= ord(c) <= 0x28FF or c in " \n" for c in text)


# ── liblouis translation ─────────────────────────────────────────────────────

def test_grade_1_french_produces_unicode_braille(liblouis):
    out = BrailleTranslator(BrailleGrade.GRADE_1).translate("bonjour la classe")
    assert out and _is_unicode_braille(out)
    # Uncontracted: plain letters map to their standard cells.
    assert out.startswith("⠃⠕⠝⠚⠕⠥⠗")


def test_grade_2_contracts_where_grade_1_does_not(liblouis):
    sentence = "les enfants de la classe travaillent pour comprendre les fractions"
    grade_1 = BrailleTranslator(BrailleGrade.GRADE_1).translate(sentence)
    grade_2 = BrailleTranslator(BrailleGrade.GRADE_2).translate(sentence)
    assert _is_unicode_braille(grade_2)
    assert len(grade_2) < len(grade_1)


def test_embosser_encoding_is_braille_ascii(liblouis):
    out = BrailleTranslator(BrailleGrade.GRADE_1, BrailleEncoding.EMBOSSER).translate("abc")
    assert out == "ABC"
    assert set(out) <= set(BRAILLE_ASCII)


def test_line_breaks_are_kept(liblouis):
    out = BrailleTranslator().translate("un\n\ndeux")
    assert out.count("\n") == 2


def test_accented_and_long_text_is_fully_translated(liblouis):
    """The output buffer grows: nothing may be silently cut off."""
    text = "élève à l'école " * 40
    out = BrailleTranslator().translate(text.strip())
    assert _is_unicode_braille(out)
    assert out.count("⠀") + out.count(" ") >= 100, "every word boundary survived"


def test_english_tables_are_available(liblouis):
    assert _is_unicode_braille(BrailleTranslator(BrailleGrade.GRADE_2, language="en").translate("the knowledge"))


def test_an_unknown_language_is_refused():
    with pytest.raises(ValueError, match="No braille table"):
        BrailleTranslator(language="xx")


# ── Braille ASCII mapping (no liblouis needed) ───────────────────────────────

def test_unicode_cells_map_to_north_american_braille_ascii():
    assert unicode_to_braille_ascii("⠁⠃⠉") == "ABC"          # dots 1, 12, 14
    assert unicode_to_braille_ascii("⠀⠼⠿") == " #="           # blank, 3456, 123456
    assert len(BRAILLE_ASCII) == 64


def test_eight_dot_cells_are_refused_for_embossing():
    with pytest.raises(BrailleEncodingError, match="dots 7-8"):
        unicode_to_braille_ascii(chr(0x2800 + 0x41))


# ── embosser ─────────────────────────────────────────────────────────────────

def test_the_simulator_wraps_words_and_paginates():
    embosser = SimulatedEmbosser(cells_per_line=10, lines_per_page=2)
    job = embosser.emboss("AAAA BBBB CCCC DDDD EEEE")
    assert [page.lines for page in job.pages] == [("AAAA BBBB", "CCCC DDDD"), ("EEEE",)]
    assert embosser.jobs == [job]


def test_a_word_longer_than_a_line_is_split():
    job = SimulatedEmbosser(cells_per_line=4, lines_per_page=10).emboss("ABCDEFGHIJ")
    assert job.pages[0].lines == ("ABCD", "EFGH", "IJ")


def test_the_simulator_writes_a_brf_file(tmp_path):
    embosser = SimulatedEmbosser(cells_per_line=10, lines_per_page=1, output_dir=str(tmp_path))
    job = embosser.emboss("AAAA\nBBBB")
    assert (tmp_path / f"{job.job_id}.brf").read_text() == "AAAA\fBBBB"


def test_the_embosser_refuses_text_that_is_not_braille_ascii():
    with pytest.raises(EmbosserError, match="Not Braille ASCII"):
        SimulatedEmbosser().emboss("bonjour")   # lowercase: print text, not Braille ASCII


def test_the_network_embosser_is_an_explicit_stub():
    with pytest.raises(NotImplementedError, match="stub"):
        NetworkEmbosser("192.0.2.10").emboss("ABC")


def test_translation_to_embosser_end_to_end(liblouis, tmp_path):
    ascii_braille = BrailleTranslator(BrailleGrade.GRADE_2, BrailleEncoding.EMBOSSER).translate(
        "Les fractions servent à partager.\nSources :\n1. Wikipédia"
    )
    job = SimulatedEmbosser(output_dir=str(tmp_path)).emboss(ascii_braille)
    assert job.pages and all(len(line) <= 40 for page in job.pages for line in page.lines)
