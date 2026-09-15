"""Local embedding wrapper, kept off Nebius deliberately.

The hackathon rule only requires the model *inference* to run on Nebius / an NVIDIA
open source model. The embedder stays local because it sits on the hot path for every
retrieval, and the whole point of this architecture is to keep working on constrained,
low-connectivity hardware. Swap LOCAL_EMBEDDING_MODEL in apu/config.py for a different
size depending on the target device; nothing else in the pipeline needs to change.

The TextEmbedding wiring is ported from Akili (llm_provider.LocalOnnxEmbedder and
get_embedder, plus embedding_config.normalize_vector). Two behaviours come with it:

  - the model is resolved from LOCAL_EMBEDDING_CACHE_DIR and never downloaded at
    runtime. On a device with no connectivity a download attempt hangs at the first
    question instead of failing. Populate the cache once, on a connected machine, with
    scripts/fetch_embedding_model.py.
  - the model is built on first use and cached for the process, not at import. Built
    at import, it downloaded weights as a side effect of `import` and made every
    module that imports this one unusable offline.

Not ported: Akili's EMBEDDING_PROVIDER=google remote fallback, since the embedder is
meant to stay local. The scaffold also mentioned an explicit ONNX CPUExecutionProvider
setup; the Akili code never passes `providers=` and relies on fastembed's default, so
none is set here either.
"""

import asyncio
import math
import os
import threading
from typing import List, Sequence

from fastembed import TextEmbedding

from apu import config
from apu.logger import get_logger

logger = get_logger(__name__)


class LocalOnnxEmbedder:
    """
    ONNX embedder via fastembed. No torch, no sentence-transformers: both are
    disqualifying on the target hardware.
    """

    def __init__(self, model_name: str, cache_dir: str, expected_dim: int,
                 allow_download: bool = False):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.expected_dim = expected_dim

        if allow_download:
            # Only scripts/fetch_embedding_model.py takes this path: it runs on a
            # connected machine by definition.
            os.makedirs(cache_dir, exist_ok=True)
            self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        else:
            if not os.path.isdir(cache_dir):
                raise RuntimeError(_missing_model_message(model_name, cache_dir))
            try:
                self._model = TextEmbedding(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    _missing_model_message(model_name, cache_dir)
                ) from e

        logger.info(
            "Embedder: local ONNX — model=%s dim=%d cache=%s",
            model_name, expected_dim, cache_dir,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        # fastembed is synchronous CPU work; keep it off the event loop.
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def _missing_model_message(model_name: str, cache_dir: str) -> str:
    return (
        f"Embedding model '{model_name}' is not present in {cache_dir}.\n"
        f"The local embedder never downloads at runtime, because on a machine "
        f"with no connectivity that hangs instead of failing.\n"
        f"Fix: on a CONNECTED machine run\n"
        f"    python scripts/fetch_embedding_model.py\n"
        f"then copy '{cache_dir}' to this device, or point LOCAL_EMBEDDING_CACHE_DIR "
        f"at an existing cache."
    )


_embedder = None
_embedder_lock = threading.Lock()


def get_embedder():
    """
    Return the process-wide embedder.

    Cached: the model is ~220MB of ONNX weights, which must load once per process,
    not once per memory write-back.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is None:
            _embedder = build_embedder()
    return _embedder


def build_embedder(allow_download: bool = False) -> LocalOnnxEmbedder:
    """
    Uncached embedder factory.

    `allow_download=True` is for the fetch script, which runs on a connected
    machine. The runtime must never use it.
    """
    return LocalOnnxEmbedder(
        model_name=config.LOCAL_EMBEDDING_MODEL,
        cache_dir=config.LOCAL_EMBEDDING_CACHE_DIR,
        expected_dim=config.LOCAL_EMBEDDING_DIM,
        allow_download=allow_download,
    )


def reset_embedder() -> None:
    """Drop the cached embedder. For tests and for config changes at runtime."""
    global _embedder
    with _embedder_lock:
        _embedder = None


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, used for both block matching and query retrieval."""
    return get_embedder().embed_documents(texts)


def normalize_vector(vector: Sequence[float]) -> List[float]:
    """
    Scale a vector to unit L2 length.

    Part of the embedding contract, not a storage detail: the L3 certainty scale is
    `1 - distance/2`, and with LanceDB's squared-L2 metric that equals cosine
    similarity only for unit vectors. Queries and stored vectors must both be
    normalised or the thresholds in apu.config are meaningless.

    A zero vector has no direction and is returned unchanged rather than raising.
    """
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0.0:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]
