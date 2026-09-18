"""
The real ONNX embedder — semantics, not plumbing.

Target: apu/embeddings/local_embedder.py

The semantic tests skip when the model cache is absent, so the suite still runs on
a machine that has not fetched the 240MB model. Point LOCAL_EMBEDDING_CACHE_DIR at
an existing cache (or run scripts/fetch_embedding_model.py) to exercise them.
"""

import pytest

from apu import config
from apu.embeddings import local_embedder


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_a_paraphrase_ranks_above_an_unrelated_sentence(real_local_embedder):
    """
    Catches a broken tokenizer or the wrong pooling config, which does not raise:
    it just produces mediocre results that look like the model being weak.
    """
    vectors = await real_local_embedder.aembed_documents([
        "How do you add two fractions?",
        "What is the way to compute the sum of two fractions?",
        "The cat is sleeping on the rug by the fireplace.",
    ])
    q, para, other = (local_embedder.normalize_vector(v) for v in vectors)

    sim_paraphrase = _cosine(q, para)
    sim_unrelated = _cosine(q, other)

    assert sim_paraphrase > sim_unrelated, (
        f"paraphrase {sim_paraphrase:.3f} did not outrank unrelated "
        f"{sim_unrelated:.3f} -- suspect tokenizer or pooling"
    )
    assert sim_paraphrase > 0.6, f"paraphrase similarity only {sim_paraphrase:.3f}"


async def test_cross_lingual_pairs_are_closer_than_unrelated(real_local_embedder):
    """The model is multilingual; the curriculum mixes French and English."""
    vectors = await real_local_embedder.aembed_documents([
        "The student is learning fractions.",
        "L'élève apprend les fractions.",
        "The weather is cold today.",
    ])
    en, fr, other = (local_embedder.normalize_vector(v) for v in vectors)
    assert _cosine(en, fr) > _cosine(en, other)


async def test_the_embedder_produces_the_configured_dimension(real_local_embedder):
    vector = await real_local_embedder.aembed_query("check")
    assert len(vector) == config.LOCAL_EMBEDDING_DIM


async def test_embedding_is_deterministic(real_local_embedder):
    a = await real_local_embedder.aembed_query("negative numbers")
    b = await real_local_embedder.aembed_query("negative numbers")
    assert a == pytest.approx(b)


def test_the_scaffold_embed_entry_point_uses_the_cached_embedder(real_local_embedder):
    """embed() is kept from the scaffold as the synchronous batch entry point."""
    vectors = local_embedder.embed(["one", "two"])
    assert len(vectors) == 2
    assert all(len(v) == config.LOCAL_EMBEDDING_DIM for v in vectors)


def test_the_configured_model_id_is_one_fastembed_accepts():
    """
    The scaffold's original default, "paraphrase-multilingual-MiniLM-L12-v2",
    is not a fastembed id and could never load.
    """
    from fastembed import TextEmbedding

    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    assert config.LOCAL_EMBEDDING_MODEL in supported
    assert "paraphrase-multilingual-MiniLM-L12-v2" not in supported


def test_importing_the_module_does_not_load_a_model():
    """Built at import, the model would download as a side effect of `import`."""
    local_embedder.reset_embedder()
    assert local_embedder._embedder is None


def test_the_runtime_embedder_never_downloads():
    """A download attempt on a disconnected device hangs instead of failing."""
    with pytest.raises(RuntimeError, match="not present in"):
        local_embedder.LocalOnnxEmbedder(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="/nonexistent/cache/dir",
            expected_dim=384,
        )


def test_the_missing_model_message_is_actionable():
    msg = local_embedder._missing_model_message("some/model", "/some/cache")
    assert "scripts/fetch_embedding_model.py" in msg
    assert "LOCAL_EMBEDDING_CACHE_DIR" in msg


def test_normalize_vector_leaves_a_zero_vector_alone():
    assert local_embedder.normalize_vector([0.0, 0.0]) == [0.0, 0.0]
