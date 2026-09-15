"""
Dynamic-block cap enforcement and LRU eviction policy.

Target: apu/mmu/dll.py (create_dynamic_block), apu/config.py (MAX_DYNAMIC_BLOCKS)
"""

import pytest

from apu import config
from apu.mmu import cache_l1
from apu.mmu import dll as mmu
from tests.conftest import V_A, check_dll_invariants


async def test_max_dynamic_blocks_is_akilis_five():
    """
    Pins the value ported from Akili. The scaffold documents 12 active blocks
    (config.MAX_ACTIVE_BLOCKS, from the travel-agent APU); that conflict is open,
    see HACKATHON.md. Change this test together with the decision.
    """
    assert config.MAX_DYNAMIC_BLOCKS == 5


async def test_init_dll_carries_the_cap_into_the_state(akili_paths):
    dll = await mmu.init_dll()
    assert dll["dynamic_block_max"] == config.MAX_DYNAMIC_BLOCKS
    assert dll["dynamic_block_count"] == 0


async def test_creating_blocks_up_to_the_cap_stays_healthy(akili_paths):
    dll = await mmu.init_dll()
    for i in range(config.MAX_DYNAMIC_BLOCKS):
        dll = await mmu.create_dynamic_block(
            block_id=f"dyn_{i}",
            label=f"Dyn {i}",
            block_type="temp",
            initial_content=f"content {i}",
            keywords=["k"],
            created_by="test",
            dll=dll,
            vector=list(V_A),
        )
    assert dll["dynamic_block_count"] == 5
    assert check_dll_invariants(dll) == []
    # 4 fixed + 5 dynamic
    assert len(dll["nodes"]) == 9


async def test_duplicate_block_id_is_rejected(akili_paths):
    dll = await mmu.init_dll()
    dll = await mmu.create_dynamic_block(
        "dup", "Dup", "temp", "c", [], "test", dll, vector=list(V_A)
    )
    with pytest.raises(ValueError, match="already exists"):
        await mmu.create_dynamic_block(
            "dup", "Dup", "temp", "c", [], "test", dll, vector=list(V_A)
        )


async def test_a_block_without_a_vector_is_refused(akili_paths):
    dll = await mmu.init_dll()
    with pytest.raises(ValueError, match="no vector"):
        await mmu.create_dynamic_block("x", "X", "temp", "c", [], "test", dll)


async def test_eviction_of_a_single_dynamic_block_at_the_cap(akili_paths):
    dll = await mmu.init_dll()
    dll["dynamic_block_max"] = 1
    dll = await mmu.create_dynamic_block(
        "first", "First", "temp", "c1", [], "test", dll, vector=list(V_A)
    )
    assert dll["dynamic_block_count"] == 1

    dll = await mmu.create_dynamic_block(
        "second", "Second", "temp", "c2", [], "test", dll, vector=list(V_A)
    )
    assert "first" not in dll["nodes"], "the older block should have been paged out"
    assert "second" in dll["nodes"]
    assert dll["dynamic_block_count"] == 1
    assert check_dll_invariants(dll) == []


async def test_working_set_stays_bounded_past_the_cap(akili_paths):
    dll = await mmu.init_dll()
    for i in range(config.MAX_DYNAMIC_BLOCKS + 3):
        dll = await mmu.create_dynamic_block(
            block_id=f"dyn_{i}",
            label=f"Dyn {i}",
            block_type="temp",
            initial_content=f"content {i}",
            keywords=["k"],
            created_by="test",
            dll=dll,
            vector=list(V_A),
        )
    assert dll["dynamic_block_count"] <= config.MAX_DYNAMIC_BLOCKS
    assert check_dll_invariants(dll) == []


async def test_fixed_blocks_are_never_evicted(akili_paths):
    dll = await mmu.init_dll()
    for i in range(config.MAX_DYNAMIC_BLOCKS + 3):
        dll = await mmu.create_dynamic_block(
            f"dyn_{i}", f"Dyn {i}", "temp", f"c{i}", [], "test", dll, vector=list(V_A)
        )
    for fixed_id in ("current_session", "active_course",
                     "learning_preferences", "student_profile"):
        assert fixed_id in dll["nodes"]


async def test_a_paged_out_block_stays_in_l3(akili_paths):
    """Page-out leaves the DLL, not the store: the row is still in user_memory."""
    from apu.storage import lance_driver

    dll = await mmu.init_dll()
    dll["dynamic_block_max"] = 1
    dll = await mmu.create_dynamic_block(
        "first", "First", "temp", "kept in L3", [], "test", dll, vector=list(V_A)
    )
    dll = await mmu.create_dynamic_block(
        "second", "Second", "temp", "c2", [], "test", dll, vector=list(V_A)
    )
    assert "first" not in dll["nodes"]
    assert await lance_driver.get_block_content("first") == "kept in L3"


async def test_eviction_picks_the_least_recently_accessed_block(
    akili_paths, stub_embeddings
):
    dll = await mmu.init_dll()
    dll["dynamic_block_max"] = 2
    for i in ("old", "new"):
        dll = await mmu.create_dynamic_block(
            i, i, "temp", f"c-{i}", [], "test", dll, vector=list(V_A)
        )

    # "Access" the old block through every read path the app actually has.
    cache_l1.set("old", "c-old", block_type="temp")
    cache_l1.get("old")
    dll = await mmu.update_node_content("old", "touched", dll)

    assert dll["nodes"]["old"]["last_accessed"] is not None, (
        "reading a block never records an access time"
    )

    dll = await mmu.create_dynamic_block(
        "third", "third", "temp", "c3", [], "test", dll, vector=list(V_A)
    )
    assert "new" not in dll["nodes"], "the untouched block should be the victim"
    assert "old" in dll["nodes"]
