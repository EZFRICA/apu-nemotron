"""Plain text for places where Markdown and math delimiters are noise (notebook entries)."""

import re

_MARKDOWN_NOISE = re.compile(r"(\*\*|__|`|^#{1,6}\s*|\\[()\[\]]|\$)", re.M)


def plain_text(text: str) -> str:
    """Strip Markdown emphasis, headings, code ticks and LaTeX delimiters."""
    return _MARKDOWN_NOISE.sub("", text)
