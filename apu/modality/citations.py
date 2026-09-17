"""Rendering web-search sources with an answer, according to the output modality.

A search is never invisible: every answer built from web results carries its sources.
How they are carried depends on the channel:

  - text or braille: a numbered list (title and URL) at the end of the answer. For braille
    the whole written text, list included, is then translated by the braille translator.
  - voice: the source names are said aloud ("D'après Wikipédia...") and URLs are never
    read out, since a spoken URL is noise. If the session also has a text display, the
    written list is provided alongside the spoken answer.
"""

from dataclasses import dataclass
from urllib.parse import urlparse

from apu.modality.mode import InteractionMode

# Display names for sites whose domain does not read well aloud. Anything else falls back
# to its capitalised second-level domain ("lumni.fr" -> "Lumni").
_KNOWN_SITE_NAMES = {
    "wikipedia.org": "Wikipédia",
    "wikimedia.org": "Wikimedia",
    "britannica.com": "Britannica",
    "larousse.fr": "Larousse",
    "khanacademy.org": "Khan Academy",
    "lumni.fr": "Lumni",
    "education.gouv.fr": "le ministère de l'Éducation nationale",
}


@dataclass(frozen=True)
class Source:
    title: str
    url: str

    @property
    def site_name(self) -> str:
        host = (urlparse(self.url).hostname or "").lower()
        host = host.removeprefix("www.")
        for domain, name in _KNOWN_SITE_NAMES.items():
            if host == domain or host.endswith("." + domain):
                return name
        labels = [label for label in host.split(".") if label]
        if len(labels) >= 2:
            return labels[-2].capitalize()
        return self.title or host or self.url


@dataclass(frozen=True)
class RenderedAnswer:
    """What each channel receives. None means the channel gets nothing."""

    spoken: str | None
    written: str | None


def written_source_list(sources: list[Source]) -> str:
    lines = ["Sources :"]
    for number, source in enumerate(sources, start=1):
        lines.append(f"{number}. {source.title} — {source.url}")
    return "\n".join(lines)


def spoken_attribution(sources: list[Source]) -> str:
    # Each site named once, in the order results came back: three Wikipedia pages are
    # still "d'après Wikipédia".
    names: list[str] = []
    for source in sources:
        if source.site_name not in names:
            names.append(source.site_name)
    if len(names) == 1:
        joined = names[0]
    else:
        joined = ", ".join(names[:-1]) + " et " + names[-1]
    return f"D'après {joined}"


def render_answer(answer: str, sources: list[Source], mode: InteractionMode) -> RenderedAnswer:
    if mode.is_spoken:
        if not sources:
            return RenderedAnswer(
                spoken=answer, written=answer if mode.text_display_available else None
            )
        written = (
            f"{answer}\n\n{written_source_list(sources)}" if mode.text_display_available else None
        )
        return RenderedAnswer(spoken=f"{spoken_attribution(sources)} : {answer}", written=written)

    if not sources:
        return RenderedAnswer(spoken=None, written=answer)
    return RenderedAnswer(spoken=None, written=f"{answer}\n\n{written_source_list(sources)}")
