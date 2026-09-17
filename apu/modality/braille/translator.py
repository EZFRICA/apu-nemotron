"""Print text to braille with liblouis: grade 1 or grade 2, Unicode braille or embosser encoding.

  - Grade 1 is uncontracted braille (French "intégral"), grade 2 is contracted ("abrégé").
  - Unicode output (U+2800 block) is for refreshable braille displays and screen readers.
  - Embosser output is North American Braille ASCII, the 6-dot encoding of .brf files and
    of most embossers. It is derived here from the dot patterns rather than from a liblouis
    display table, so it does not depend on which display tables a system happens to ship.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from apu.modality.braille import _liblouis


class BrailleGrade(IntEnum):
    GRADE_1 = 1
    GRADE_2 = 2


class BrailleEncoding(StrEnum):
    UNICODE = "unicode"
    EMBOSSER = "embosser"


# (language, grade) -> liblouis table. French tables are BFU (braille français unifié):
# fr-bfu-comp6 is declared literary, uncontracted, 6 dots; fr-bfu-g2 is contracted grade 2.
TABLES: dict[tuple[str, BrailleGrade], str] = {
    ("fr", BrailleGrade.GRADE_1): "fr-bfu-comp6.utb",
    ("fr", BrailleGrade.GRADE_2): "fr-bfu-g2.ctb",
    ("en", BrailleGrade.GRADE_1): "en-ueb-g1.ctb",
    ("en", BrailleGrade.GRADE_2): "en-ueb-g2.ctb",
}

# North American Braille ASCII, indexed by the 6-dot cell bitmask (dot 1 = bit 0 ... dot 6 =
# bit 5). Index 0 is the blank cell.
BRAILLE_ASCII = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="

_UNICODE_BRAILLE_BASE = 0x2800


class BrailleEncodingError(ValueError):
    """A cell cannot be expressed in the requested encoding."""


def unicode_to_braille_ascii(unicode_braille: str) -> str:
    characters = []
    for character in unicode_braille:
        code_point = ord(character)
        if character in "\n\f":
            characters.append(character)
            continue
        if character == " ":
            characters.append(" ")
            continue
        if not _UNICODE_BRAILLE_BASE <= code_point <= _UNICODE_BRAILLE_BASE + 0xFF:
            raise BrailleEncodingError(f"Not a braille cell: {character!r}")
        dots = code_point - _UNICODE_BRAILLE_BASE
        if dots > 0x3F:
            # Dots 7 and 8 have no Braille ASCII representation; an embosser would silently
            # drop them, so refuse instead.
            raise BrailleEncodingError(
                f"Cell {character!r} uses dots 7-8, which 6-dot embosser encoding cannot represent."
            )
        characters.append(BRAILLE_ASCII[dots])
    return "".join(characters)


@dataclass(frozen=True)
class BrailleTranslator:
    grade: BrailleGrade = BrailleGrade.GRADE_1
    encoding: BrailleEncoding = BrailleEncoding.UNICODE
    language: str = "fr"

    def __post_init__(self) -> None:
        object.__setattr__(self, "grade", BrailleGrade(self.grade))
        object.__setattr__(self, "encoding", BrailleEncoding(self.encoding))
        if (self.language, self.grade) not in TABLES:
            raise ValueError(
                f"No braille table for language={self.language!r} grade={int(self.grade)}."
            )

    @property
    def table(self) -> str:
        return TABLES[(self.language, self.grade)]

    def translate(self, text: str) -> str:
        # Line by line: line breaks are layout the reader relies on (a source list, an
        # exercise statement), and translating a whole paragraph block would reflow them.
        braille_lines = [
            _liblouis.translate_string(self.table, line, _liblouis.DOTS_IO | _liblouis.UC_BRL)
            if line
            else ""
            for line in text.split("\n")
        ]
        unicode_braille = "\n".join(braille_lines)
        if self.encoding is BrailleEncoding.EMBOSSER:
            return unicode_to_braille_ascii(unicode_braille)
        return unicode_braille
