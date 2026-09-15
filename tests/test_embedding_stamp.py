"""
The guard against a silent vector-space mismatch.

Target: apu/storage/lance_driver.py (verify_embedding_space, read/write_stamp)

A mismatch does not fail on its own: it returns confidently ranked nonsense.
These tests assert it becomes a readable refusal instead. Akili's two tests on the
cloud pipeline sharing the client's config are not ported with the pipeline.
"""

import json

import pytest

from apu import config
from apu.storage import lance_driver
from tests.conftest import DIM, V_A


def _row(rid, vector, content="c"):
    return {
        "id": rid, "chapter": rid, "content": content, "block_type": "cours",
        "class_level": "6eme", "subject": "math", "vector": list(vector),
        "updated_at": "2026-01-01T00:00:00",
    }


# ── the dimension check ──────────────────────────────────────────────────────

async def test_a_wider_registry_is_refused_not_searched(akili_paths, monkeypatch):
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", [0.1] * 16)])
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_DIM", 8)

    with pytest.raises(lance_driver.EmbeddingMismatch) as exc:
        await lance_driver.search_block_index([0.1] * 8, class_level="6eme",
                                              subject="math")

    msg = str(exc.value)
    assert "16-dimension" in msg
    assert "8" in msg
    assert "migrate_embeddings.py" in msg


async def test_the_refusal_names_the_configured_model(akili_paths, monkeypatch):
    db = lance_driver.get_db()
    db.create_table("user_memory", data=[_row("a", [0.1] * 16)])
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODEL", "acme/tiny-embed")

    with pytest.raises(lance_driver.EmbeddingMismatch) as exc:
        await lance_driver.search_block_index(list(V_A))

    assert "acme/tiny-embed" in str(exc.value)


async def test_a_matching_dimension_searches_normally(akili_paths):
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A)])

    results = await lance_driver.search_block_index(
        list(V_A), class_level="6eme", subject="math"
    )
    assert [r["block_id"] for r in results] == ["ch1"]


# ── the model check ──────────────────────────────────────────────────────────

async def test_a_different_model_at_the_same_dimension_is_refused(
    akili_paths, monkeypatch
):
    await lance_driver.upsert_local_block(
        block_id="student_profile", content="Marc", block_type="fondamental",
        class_level="6eme", subject="math", vector=list(V_A),
    )
    assert lance_driver.read_stamp("user_memory")["model"] == "test/stub-embedder"

    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODEL", "other/same-width-model")

    with pytest.raises(lance_driver.EmbeddingMismatch) as exc:
        await lance_driver.search_block_index(list(V_A))

    msg = str(exc.value)
    assert "test/stub-embedder" in msg
    assert "other/same-width-model" in msg
    assert "Same dimension does not mean the same vector space" in msg


# ── the stamp itself ─────────────────────────────────────────────────────────

async def test_writing_a_block_records_the_stamp(akili_paths):
    assert lance_driver.read_stamp("user_memory") is None

    await lance_driver.upsert_local_block(
        block_id="a", content="x", block_type="temp",
        class_level="6eme", subject="math", vector=list(V_A),
    )

    stamp = lance_driver.read_stamp("user_memory")
    assert stamp["model"] == "test/stub-embedder"
    assert stamp["dim"] == DIM
    assert "updated_at" in stamp


def test_a_corrupt_stamp_is_treated_as_absent(akili_paths):
    with open(config.EMBEDDING_STAMP_PATH, "w") as f:
        f.write("{not json")
    assert lance_driver.read_stamp("user_memory") is None


async def test_no_stamp_and_no_tables_is_not_an_error(akili_paths):
    assert await lance_driver.search_block_index(list(V_A)) == []


# ── stamps are per table, not per store ──────────────────────────────────────

async def test_tables_are_stamped_independently(akili_paths):
    await lance_driver.upsert_local_block(
        block_id="a", content="x", block_type="temp",
        class_level="6eme", subject="math", vector=list(V_A),
    )
    assert lance_driver.read_stamp("user_memory") is not None
    assert lance_driver.read_stamp("edu_registry") is None


async def test_a_stale_registry_is_caught_at_the_same_dimension(
    akili_paths, monkeypatch
):
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A)])
    lance_driver.write_stamp("edu_registry")

    # local memory re-embedded with a new model of the same width...
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODEL", "new/same-width")
    await lance_driver.upsert_local_block(
        block_id="a", content="x", block_type="temp",
        class_level="6eme", subject="math", vector=list(V_A),
    )
    # ...user_memory now current, edu_registry stale
    assert lance_driver.read_stamp("user_memory")["model"] == "new/same-width"
    assert lance_driver.read_stamp("edu_registry")["model"] == "test/stub-embedder"

    with pytest.raises(lance_driver.EmbeddingMismatch) as exc:
        await lance_driver.search_block_index(list(V_A), class_level="6eme",
                                              subject="math")
    assert "edu_registry" in str(exc.value)


def test_a_legacy_flat_stamp_is_honoured(akili_paths):
    with open(config.EMBEDDING_STAMP_PATH, "w") as f:
        json.dump({"model": "old/model", "dim": 8}, f)

    assert lance_driver.read_stamp("user_memory")["model"] == "old/model"
    assert lance_driver.read_stamp("edu_registry")["model"] == "old/model"
