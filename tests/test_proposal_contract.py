"""
The block-proposal contract between detector and executor.

Target: apu/core/block_detector.py (detect_new_block_opportunity)
        apu/mmu/dll.py (auto_execute_block_proposal)
"""

import pytest

from apu import config
from apu.core.block_detector import detect_new_block_opportunity
from apu.core.block_proposal import REQUIRED_FIELDS, validate
from apu.mmu import cache_l1
from apu.mmu import dll as mmu
from apu.storage import lance_driver
from tests.conftest import check_dll_invariants


def _history(n_user_msgs=3, text="Can you explain why this works?"):
    h = []
    for _ in range(n_user_msgs):
        h.append({"role": "user", "content": text})
        h.append({"role": "assistant", "content": "Sure."})
    return h


# ── the detector ─────────────────────────────────────────────────────────────

async def test_detector_returns_none_below_the_turn_threshold(akili_paths):
    dll = await mmu.init_dll()
    assert detect_new_block_opportunity(_history(1), dll) is None  # 2 entries < 4


async def test_detector_returns_none_when_the_cap_is_reached(akili_paths):
    dll = await mmu.init_dll()
    dll["dynamic_block_count"] = dll["dynamic_block_max"]
    assert detect_new_block_opportunity(_history(3), dll) is None


async def test_detector_emits_the_shared_proposal_shape(akili_paths):
    dll = await mmu.init_dll()
    proposal = detect_new_block_opportunity(_history(3), dll)

    assert proposal is not None
    assert validate(proposal) is None, validate(proposal)
    assert set(proposal.keys()) == {
        "proposed_id", "label", "type", "initial_content", "keywords", "reason",
    }
    assert proposal["proposed_id"] == "dynamic_block_1"
    assert proposal["type"] == "temp"
    assert proposal["initial_content"]
    assert all(str(proposal[f]).strip() for f in REQUIRED_FIELDS)


async def test_proposed_id_collides_after_a_page_out(akili_paths):
    """
    Known Akili behaviour, carried over: proposed_id is derived from
    dynamic_block_count, not from the ids present. page_out_block decrements the
    counter, so an id can be reused and create_dynamic_block then raises.
    """
    dll = await mmu.init_dll()
    dll["dynamic_block_count"] = 2
    first = detect_new_block_opportunity(_history(3), dll)
    assert first["proposed_id"] == "dynamic_block_3"

    dll["dynamic_block_count"] = 1
    second = detect_new_block_opportunity(_history(3), dll)
    assert second["proposed_id"] == "dynamic_block_2"


def test_trigger_list_matches_substrings_inside_unrelated_words():
    """Known Akili behaviour: 'how' matches inside 'show', 'somehow'."""
    dll = {"dynamic_block_count": 0, "dynamic_block_max": 5}
    h = [
        {"role": "user", "content": "Show me the shower schedule somehow"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Show me again"},
        {"role": "assistant", "content": "ok"},
    ]
    assert detect_new_block_opportunity(h, dll) is not None


# ── the executor ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad,why", [
    ({}, "empty"),
    (None, "none"),
    ({"proposed_id": "x", "label": "L", "initial_content": "c"}, "no type"),
    ({"proposed_id": "x", "label": "L", "type": "temp"}, "no content"),
    ({"proposed_id": "x", "label": "L", "type": "temp",
      "initial_content": "   "}, "blank content"),
    ({"label": "L", "type": "temp", "initial_content": "c"}, "no id"),
])
async def test_a_malformed_proposal_is_refused_and_writes_nothing(
    akili_paths, stub_embeddings, bad, why
):
    await mmu.init_dll()
    assert await mmu.auto_execute_block_proposal(bad) is False, why

    fresh = await mmu.load_dll()
    assert "dynamic_block_1" not in fresh["nodes"]
    assert fresh["dynamic_block_count"] == 0
    assert "user_memory" not in lance_driver.list_table_names()


async def test_the_row_that_lands_in_lancedb_carries_real_content(
    akili_paths, stub_embeddings
):
    dll = await mmu.init_dll()
    proposal = detect_new_block_opportunity(_history(3), dll)
    assert await mmu.auto_execute_block_proposal(proposal) is True

    df = lance_driver.get_db().open_table("user_memory").to_pandas()
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["id"] == "dynamic_block_1"
    assert row["content"] == proposal["initial_content"]
    assert row["block_type"] == "temp"
    # class_level/subject are hardcoded to "local" in create_dynamic_block (Akili)
    assert row["class_level"] == "local"
    assert row["subject"] == "local"
    assert len(row["vector"]) == config.LOCAL_EMBEDDING_DIM
    assert any(float(x) != 0.0 for x in row["vector"])


async def test_the_typeless_node_matches_no_ttl_and_no_threshold(akili_paths):
    """type=None falls through every lookup table in the system."""
    assert cache_l1._TTL_BY_TYPE.get(None or "", cache_l1._TTL_DEFAULT) == 300
    assert mmu.CERTAINTY_THRESHOLDS.get(None, mmu.MIN_RELEVANCE_CERTAINTY) == 0.45


async def test_a_correctly_shaped_proposal_does_land(akili_paths, stub_embeddings):
    await mmu.init_dll()
    ok = await mmu.auto_execute_block_proposal({
        "proposed_id": "dynamic_block_1",
        "label": "Topic currently being learned",
        "type": "temp",
        "initial_content": "The student is learning fractions.",
        "keywords": ["fractions"],
    })
    assert ok is True

    fresh = await mmu.load_dll()
    assert "dynamic_block_1" in fresh["nodes"]
    assert fresh["dynamic_block_count"] == 1
    assert check_dll_invariants(fresh) == []

    df = lance_driver.get_db().open_table("user_memory").to_pandas()
    assert df.shape[0] == 1
    assert df.iloc[0]["content"] == "The student is learning fractions."
    assert len(df.iloc[0]["vector"]) == config.LOCAL_EMBEDDING_DIM
    assert any(float(x) != 0.0 for x in df.iloc[0]["vector"])


async def test_a_detector_proposal_lands_in_the_dll_and_lancedb(akili_paths, stub_embeddings):
    dll = await mmu.init_dll()
    proposal = detect_new_block_opportunity(_history(3), dll)

    assert await mmu.auto_execute_block_proposal(proposal) is True

    fresh = await mmu.load_dll()
    assert "dynamic_block_1" in fresh["nodes"]
    assert fresh["nodes"]["dynamic_block_1"]["type"] == "temp"
    assert lance_driver.get_db().open_table("user_memory").to_pandas().shape[0] == 1


async def test_a_proposal_is_refused_when_the_embedder_is_unavailable(
    akili_paths, monkeypatch
):
    from apu.embeddings import local_embedder

    class _Broken:
        async def aembed_query(self, text):
            raise RuntimeError("model missing")

    monkeypatch.setattr(local_embedder, "_embedder", _Broken())
    await mmu.init_dll()
    assert await mmu.auto_execute_block_proposal({
        "proposed_id": "dynamic_block_1", "label": "L", "type": "temp",
        "initial_content": "The student is learning fractions.", "keywords": [],
    }) is False
    assert "dynamic_block_1" not in (await mmu.load_dll())["nodes"]
