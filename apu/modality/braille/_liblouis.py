"""Minimal ctypes binding to the liblouis C library.

liblouis ships its Python bindings with its C sources, not on PyPI (the PyPI package named
"louis" is unrelated), so a uv-managed environment cannot install them. The C library comes
from the system instead:

    brew install liblouis          # macOS
    apt install liblouis20         # Debian / Ubuntu

Only what the translator needs is bound. Set LIBLOUIS_PATH to point at a library in a
non-standard location.
"""

import ctypes
import ctypes.util
import os

# From liblouis.h (enum translationModes).
DOTS_IO = 4   # output braille as dot patterns rather than display-table characters
UC_BRL = 64   # with DOTS_IO: express those dot patterns as Unicode braille (U+2800 block)

_CANDIDATE_PATHS = (
    "/opt/homebrew/lib/liblouis.dylib",
    "/usr/local/lib/liblouis.dylib",
    "/usr/lib/x86_64-linux-gnu/liblouis.so.20",
    "/usr/lib/aarch64-linux-gnu/liblouis.so.20",
)

_library = None


class LiblouisUnavailable(RuntimeError):
    """The liblouis C library could not be found or loaded."""


class LiblouisTranslationError(RuntimeError):
    """liblouis refused the translation (typically: unknown or broken table)."""


def load_library():
    global _library
    if _library is not None:
        return _library

    path = (
        os.environ.get("LIBLOUIS_PATH")
        or ctypes.util.find_library("louis")
        or next((candidate for candidate in _CANDIDATE_PATHS if os.path.exists(candidate)), None)
    )
    if not path:
        raise LiblouisUnavailable(
            "liblouis is not installed. Install the C library "
            "(`brew install liblouis` on macOS, `apt install liblouis20` on Debian/Ubuntu), "
            "or set LIBLOUIS_PATH to its location."
        )
    try:
        library = ctypes.CDLL(path)
    except OSError as error:
        raise LiblouisUnavailable(f"Could not load liblouis from {path}: {error}") from error

    library.lou_version.restype = ctypes.c_char_p
    library.lou_charSize.restype = ctypes.c_int
    library.lou_translateString.restype = ctypes.c_int
    library.lou_translateString.argtypes = [
        ctypes.c_char_p,               # tableList
        ctypes.c_void_p,               # const widechar *inbuf
        ctypes.POINTER(ctypes.c_int),  # int *inlen (in: length, out: consumed)
        ctypes.c_void_p,               # widechar *outbuf
        ctypes.POINTER(ctypes.c_int),  # int *outlen (in: capacity, out: produced)
        ctypes.c_void_p,               # formtype *typeform
        ctypes.c_char_p,               # char *spacing
        ctypes.c_int,                  # int mode
    ]
    _library = library
    return library


def version() -> str:
    return load_library().lou_version().decode()


def translate_string(table_list: str, text: str, mode: int) -> str:
    library = load_library()
    # widechar is 2 or 4 bytes depending on how liblouis was built; the text has to be
    # encoded to match or every character after the first is garbage.
    char_size = library.lou_charSize()
    codec = {2: "utf-16-le", 4: "utf-32-le"}.get(char_size)
    if codec is None:
        raise LiblouisUnavailable(f"Unsupported liblouis widechar size: {char_size}")

    encoded = text.encode(codec)
    input_length = len(encoded) // char_size
    input_buffer = ctypes.create_string_buffer(encoded, len(encoded))

    # Uncontracted braille can be longer than the print text (capital and number signs),
    # so start with room to spare and grow if liblouis stops before consuming the input.
    capacity = max(input_length * 4, 64)
    while True:
        output_buffer = ctypes.create_string_buffer(capacity * char_size)
        consumed = ctypes.c_int(input_length)
        produced = ctypes.c_int(capacity)
        succeeded = library.lou_translateString(
            table_list.encode(), input_buffer, ctypes.byref(consumed),
            output_buffer, ctypes.byref(produced), None, None, mode,
        )
        if not succeeded:
            raise LiblouisTranslationError(
                f"liblouis could not translate with tables {table_list!r}."
            )
        if consumed.value >= input_length:
            break
        capacity *= 2

    return output_buffer.raw[: produced.value * char_size].decode(codec)
