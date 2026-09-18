"""Rendering web-search sources with an answer.

A search is never invisible: every answer built from web results carries its sources as a
numbered list (title and URL) at the end of the answer.
"""

from dataclasses import dataclass
from urllib.parse import urlparse

# Display names for sites whose domain does not read well. Anything else falls back to its
# capitalised second-level domain ("lumni.fr" -> "Lumni").
_KNOWN_SITE_NAMES = {
    "wikipedia.org": "Wikipedia",
    "wikimedia.org": "Wikimedia",
    "britannica.com": "Britannica",
    "larousse.fr": "Larousse",
    "khanacademy.org": "Khan Academy",
    "lumni.fr": "Lumni",
    "education.gouv.fr": "the French Ministry of Education",
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


def written_source_list(sources: list[Source]) -> str:
    lines = ["Sources:"]
    for number, source in enumerate(sources, start=1):
        lines.append(f"{number}. {source.title} — {source.url}")
    return "\n".join(lines)


def render_answer(answer: str, sources: list[Source]) -> str:
    """The answer as the student reads it: the text, then its sources when there are any."""
    if not sources:
        return answer
    return f"{answer}\n\n{written_source_list(sources)}"
