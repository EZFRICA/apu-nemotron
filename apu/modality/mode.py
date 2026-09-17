"""Interaction modes: the channel a student uses to ask, and the one the tutor answers on.

Only five input/output combinations are supported, each tied to a concrete use. Any other
pairing is refused when the mode is built, so an unsupported setup fails when a session
opens rather than halfway through rendering an answer.
"""

from dataclasses import dataclass
from enum import StrEnum


class InputChannel(StrEnum):
    TEXT = "text"        # keyboard
    VOICE = "voice"      # speech, transcribed before it reaches the tutor
    BRAILLE = "braille"  # braille keyboard


class OutputChannel(StrEnum):
    TEXT = "text"        # screen
    VOICE = "voice"      # speech synthesis
    BRAILLE = "braille"  # refreshable braille display or embosser


SUPPORTED_COMBINATIONS: frozenset[tuple[InputChannel, OutputChannel]] = frozenset({
    (InputChannel.TEXT, OutputChannel.TEXT),        # keyboard and screen
    (InputChannel.VOICE, OutputChannel.VOICE),      # fully spoken exchange
    (InputChannel.VOICE, OutputChannel.TEXT),       # dictation, answer read on screen
    (InputChannel.TEXT, OutputChannel.VOICE),       # typing, answer listened to
    (InputChannel.BRAILLE, OutputChannel.BRAILLE),  # braille keyboard, braille output
})


@dataclass(frozen=True)
class InteractionMode:
    input_channel: InputChannel
    output_channel: OutputChannel
    # Whether a written channel is also on hand in the session, e.g. a screen next to a
    # speaker. It decides whether a spoken answer also gets its written source list.
    # Forced to True when either channel is text: typing or reading implies a display.
    text_display_available: bool = False

    def __post_init__(self) -> None:
        input_channel = InputChannel(self.input_channel)
        output_channel = OutputChannel(self.output_channel)
        if (input_channel, output_channel) not in SUPPORTED_COMBINATIONS:
            supported = ", ".join(
                f"{i.value}->{o.value}" for i, o in sorted(SUPPORTED_COMBINATIONS)
            )
            raise ValueError(
                f"Unsupported interaction mode {input_channel.value}->{output_channel.value}. "
                f"Supported: {supported}."
            )
        object.__setattr__(self, "input_channel", input_channel)
        object.__setattr__(self, "output_channel", output_channel)
        if input_channel is InputChannel.TEXT or output_channel is OutputChannel.TEXT:
            object.__setattr__(self, "text_display_available", True)

    @property
    def is_spoken(self) -> bool:
        return self.output_channel is OutputChannel.VOICE

    @property
    def is_braille(self) -> bool:
        return self.output_channel is OutputChannel.BRAILLE
