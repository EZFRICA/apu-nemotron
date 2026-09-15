"""
DLL structural invariants across insert / move_to_front / page_out.

Target: apu/mmu/dll.py (ported from Akili's mmu/controller.py + mmu/block_factory.py)
"""

import pytest

from apu.mmu import dll as mmu
from tests.conftest import check_dll_invariants, make_chain, make_node


# ── the healthy baseline ─────────────────────────────────────────────────────

def test_freshly_built_chain_is_healthy():
    dll = make_chain(["a", "b", "c", "d"])
    assert check_dll_invariants(dll) == []


async def test_init_dll_produces_a_healthy_four_block_chain(akili_paths):
    dll = await mmu.init_dll()
    assert check_dll_invariants(dll) == []
    assert mmu._head_to_tail_order(dll) == [
        "current_session", "active_course", "learning_preferences", "student_profile",
    ]
    assert mmu._tail_to_head_order(dll) == [
        "student_profile", "learning_preferences", "active_course", "current_session",
    ]


# ── move_to_front ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("target", ["a", "b", "c", "d"])
def test_move_to_front_preserves_invariants(target):
    dll = make_chain(["a", "b", "c", "d"])
    dll = mmu.move_to_front(target, dll)
    assert check_dll_invariants(dll) == [], f"moving {target!r} broke the chain"
    assert dll["head_id"] == target
    assert set(mmu._head_to_tail_order(dll)) == {"a", "b", "c", "d"}


def test_move_to_front_of_tail_repoints_tail():
    dll = make_chain(["a", "b", "c"])
    dll = mmu.move_to_front("c", dll)
    assert dll["head_id"] == "c"
    assert dll["tail_id"] == "b"
    assert mmu._head_to_tail_order(dll) == ["c", "a", "b"]
    assert mmu._tail_to_head_order(dll) == ["b", "a", "c"]


def test_move_to_front_repeated_is_stable():
    dll = make_chain(["a", "b", "c", "d"])
    for target in ["c", "a", "d", "c", "b", "b"]:
        dll = mmu.move_to_front(target, dll)
        assert check_dll_invariants(dll) == [], f"after moving {target!r}"


def test_move_to_front_of_unknown_id_is_a_noop():
    dll = make_chain(["a", "b"])
    out = mmu.move_to_front("ghost", dll)
    assert out["head_id"] == "a"


# ── insert_node_by_type ──────────────────────────────────────────────────────

@pytest.mark.parametrize("block_type", ["temp", "fondamental", "projet", "course"])
def test_insert_node_by_type_preserves_invariants(block_type):
    dll = make_chain(["a", "b", "c"])
    node = make_node("new", node_type=block_type)
    dll = mmu.insert_node_by_type(block_type, node, dll)
    assert check_dll_invariants(dll) == [], f"inserting a {block_type!r} block"
    assert set(mmu._head_to_tail_order(dll)) == {"a", "b", "c", "new"}


def test_insert_temp_lands_at_head():
    dll = make_chain(["a", "b", "c"])
    dll = mmu.insert_node_by_type("temp", make_node("new", "temp"), dll)
    assert dll["head_id"] == "new"
    assert mmu._head_to_tail_order(dll) == ["new", "a", "b", "c"]


def test_insert_projet_lands_right_after_head():
    dll = make_chain(["a", "b", "c"])
    dll = mmu.insert_node_by_type("projet", make_node("new", "projet"), dll)
    assert mmu._head_to_tail_order(dll) == ["a", "new", "b", "c"]


def test_insert_fondamental_lands_just_before_tail_never_at_tail():
    """A dynamic 'fondamental' block can never become TAIL by insertion."""
    dll = make_chain(["a", "b", "c"])
    dll = mmu.insert_node_by_type("fondamental", make_node("new", "fondamental"), dll)
    assert mmu._head_to_tail_order(dll) == ["a", "b", "new", "c"]
    assert dll["tail_id"] == "c"


def test_insert_unknown_type_falls_through_to_the_projet_branch():
    """Any type without a branch (cours, course, ...) is inserted after HEAD."""
    dll = make_chain(["a", "b", "c"])
    dll = mmu.insert_node_by_type("course", make_node("new", "course"), dll)
    assert mmu._head_to_tail_order(dll) == ["a", "new", "b", "c"]


# ── page_out_block ───────────────────────────────────────────────────────────

async def test_page_out_middle_node(akili_paths):
    dll = make_chain(["a", "b", "c", "d"])
    dll = await mmu.page_out_block("b", dll)
    assert check_dll_invariants(dll) == []
    assert mmu._head_to_tail_order(dll) == ["a", "c", "d"]
    assert mmu._tail_to_head_order(dll) == ["d", "c", "a"]


async def test_page_out_the_current_head(akili_paths):
    dll = make_chain(["a", "b", "c"])
    dll = await mmu.page_out_block("a", dll)
    assert check_dll_invariants(dll) == []
    assert dll["head_id"] == "b"
    assert dll["nodes"]["b"]["prev"] is None
    assert mmu._head_to_tail_order(dll) == ["b", "c"]


async def test_page_out_the_current_tail(akili_paths):
    dll = make_chain(["a", "b", "c"])
    dll = await mmu.page_out_block("c", dll)
    assert check_dll_invariants(dll) == []
    assert dll["tail_id"] == "b"
    assert dll["nodes"]["b"]["next"] is None
    assert mmu._tail_to_head_order(dll) == ["b", "a"]


async def test_page_out_is_refused_for_fixed_blocks(akili_paths):
    dll = make_chain(["a", "b", "c"], fixed={"b"})
    before = mmu._head_to_tail_order(dll)
    dll = await mmu.page_out_block("b", dll)
    assert mmu._head_to_tail_order(dll) == before


async def test_page_out_unknown_id_is_a_noop(akili_paths):
    dll = make_chain(["a", "b"])
    dll = await mmu.page_out_block("nope", dll)
    assert check_dll_invariants(dll) == []


async def test_page_out_last_remaining_node_leaves_a_consistent_empty_dll(akili_paths):
    dll = make_chain(["only"])
    dll = await mmu.page_out_block("only", dll)
    assert dll["nodes"] == {}
    assert dll["head_id"] is None
    assert dll["tail_id"] is None
    assert check_dll_invariants(dll) == []


# ── mixed sequence ───────────────────────────────────────────────────────────

async def test_mixed_insert_move_pageout_sequence_keeps_the_chain_healthy(akili_paths):
    dll = make_chain(["a", "b", "c"])
    dll = mmu.insert_node_by_type("temp", make_node("t1", "temp"), dll)
    dll = mmu.move_to_front("c", dll)
    dll = mmu.insert_node_by_type("fondamental", make_node("f1", "fondamental"), dll)
    dll = await mmu.page_out_block("t1", dll)
    dll = mmu.move_to_front("f1", dll)
    dll = await mmu.page_out_block(dll["head_id"], dll)
    dll = await mmu.page_out_block(dll["tail_id"], dll)
    assert check_dll_invariants(dll) == []


# ── known defect carried over from Akili ─────────────────────────────────────

async def test_delete_block_stitching_calls_a_driver_function_that_does_not_exist(
    akili_paths,
):
    """
    Pins a latent Akili bug, ported as-is: lance_driver has no delete_local_block,
    so deleting a dynamic block raises AttributeError before touching the chain.
    Invert this test when the driver function is written.
    """
    dll = make_chain(["a", "b", "c"])
    with pytest.raises(AttributeError, match="delete_local_block"):
        await mmu.delete_block_stitching("b", dll)
