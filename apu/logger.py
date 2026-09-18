"""Central logging, imported by every module. Ported from Akili's logger.py."""

import logging
import os
import sys

from apu import config

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m",     # Gray
        logging.INFO: "\033[94m",      # Blue
        logging.WARNING: "\033[93m",   # Yellow
        logging.ERROR: "\033[91m",     # Red
        logging.CRITICAL: "\033[1;91m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def configure_root_logger():
    """Configure the root logger with the colored formatter."""
    root_logger = logging.getLogger()

    # Avoid duplicate handlers if already configured
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)

    # Reduce noise from external libraries. The openai client logs full request
    # options at DEBUG, prompts included, which does not belong in a runtime log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    # NeMo Guardrails logs every Colang event at INFO, dozens of lines per student turn,
    # which buries the tutor's own log under the rail's internals.
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)

    # File handler, no colors, under the data directory rather than the working directory:
    # it is per-device state like the rest, and it is wiped by the demo reset.
    #
    # INFO, never DEBUG: this file outlives the process and the students are minors. DEBUG
    # carries block contents and library internals, so a debug line added later would quietly
    # start writing children's schoolwork to disk. Raise it deliberately, for a session, when
    # debugging something you cannot reproduce otherwise.
    os.makedirs(config.DATA_DIR, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(config.DATA_DIR, "apu_runtime.log"), mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    clean_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(clean_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger for the module `name`."""
    configure_root_logger()
    return logging.getLogger(name.split(".")[-1])
