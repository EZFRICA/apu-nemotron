"""Embossers: an abstract interface, a working simulator, and a network embosser stub.

All embossers take North American Braille ASCII (BrailleEncoding.EMBOSSER) and lay it out
on pages of a fixed number of cells per line and lines per page, the physical limits of
the paper. The simulator does exactly that layout and keeps the result, optionally as .brf
files, so the whole output path can be exercised without hardware.
"""

import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from apu.modality.braille.translator import BRAILLE_ASCII

# 40 cells x 25 lines is the common layout for 11 x 11.5 inch braille paper.
DEFAULT_CELLS_PER_LINE = 40
DEFAULT_LINES_PER_PAGE = 25

_ALLOWED_CHARACTERS = frozenset(BRAILLE_ASCII) | {"\n"}


class EmbosserError(RuntimeError):
    """The job cannot be embossed as given."""


@dataclass(frozen=True)
class EmbossedPage:
    number: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class EmbossJob:
    job_id: str
    pages: tuple[EmbossedPage, ...]

    def to_brf(self) -> str:
        # BRF: one line per braille line, pages separated by a form feed.
        return "\f".join("\n".join(page.lines) for page in self.pages)


class Embosser(ABC):
    def __init__(
        self,
        cells_per_line: int = DEFAULT_CELLS_PER_LINE,
        lines_per_page: int = DEFAULT_LINES_PER_PAGE,
    ) -> None:
        if cells_per_line < 1 or lines_per_page < 1:
            raise ValueError("cells_per_line and lines_per_page must be positive.")
        self.cells_per_line = cells_per_line
        self.lines_per_page = lines_per_page

    @abstractmethod
    def emboss(self, braille_ascii: str) -> EmbossJob:
        """Lay out and emboss Braille ASCII text."""

    def layout(self, braille_ascii: str) -> tuple[EmbossedPage, ...]:
        unexpected = set(braille_ascii) - _ALLOWED_CHARACTERS
        if unexpected:
            raise EmbosserError(
                f"Not Braille ASCII: {sorted(unexpected)!r}. Translate with "
                "BrailleEncoding.EMBOSSER first."
            )

        lines: list[str] = []
        for paragraph in braille_ascii.split("\n"):
            lines.extend(self._wrap(paragraph))

        pages = [
            EmbossedPage(number=index // self.lines_per_page + 1,
                         lines=tuple(lines[index:index + self.lines_per_page]))
            for index in range(0, len(lines), self.lines_per_page)
        ]
        return tuple(pages) or (EmbossedPage(number=1, lines=()),)

    def _wrap(self, paragraph: str) -> list[str]:
        if not paragraph:
            return [""]
        # A space is a blank cell, so words wrap at spaces like print; a word longer than a
        # line is split, since a braille line cannot overflow the paper.
        lines: list[str] = []
        current = ""
        for word in paragraph.split(" "):
            while len(word) > self.cells_per_line:
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[: self.cells_per_line])
                word = word[self.cells_per_line:]
            candidate = f"{current} {word}" if current else word
            if len(candidate) <= self.cells_per_line:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


class SimulatedEmbosser(Embosser):
    """Performs the layout and keeps every job; writes <job_id>.brf if output_dir is set."""

    def __init__(self, *args, output_dir: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.jobs: list[EmbossJob] = []

    def emboss(self, braille_ascii: str) -> EmbossJob:
        job = EmbossJob(job_id=str(uuid.uuid4()), pages=self.layout(braille_ascii))
        self.jobs.append(job)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"{job.job_id}.brf"), "w", encoding="ascii") as brf:
                brf.write(job.to_brf())
        return job


class NetworkEmbosser(Embosser):
    """STUB. Sending jobs to a physical embosser over the network is not implemented."""

    def __init__(self, host: str, port: int = 9100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port

    def emboss(self, braille_ascii: str) -> EmbossJob:
        raise NotImplementedError(
            f"NetworkEmbosser is a stub: sending jobs to {self.host}:{self.port} is not "
            "implemented. Use SimulatedEmbosser."
        )
