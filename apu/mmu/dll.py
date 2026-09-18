"""Semantic MMU: doubly linked list of memory blocks, LRU paging, BMJ routing.

Ported from Akili, where this logic was split across app_local/mmu/controller.py
(DLL state, BMJ routing, move-to-front, traversals) and app_local/mmu/block_factory.py
(insertion by type, page-out, LRU eviction, block creation). Merged here with the
behaviour unchanged; only imports and the settings module moved.

Design invariants, as the ported code actually implements them:
  - insertion positions: temp at HEAD, projet right after HEAD, fondamental just
    before TAIL (never at TAIL). Any other type (cours, unknown) takes the projet
    branch. The four fixed blocks start as
    current_session (temp, HEAD) -> active_course (cours)
    -> learning_preferences (fondamental) -> student_profile (fondamental, TAIL).
  - move-to-front (BMJ) in search_memory: only the FIRST retrieved result that is a
    DLL node is promoted per search, not every match.
  - cap: config.MAX_DYNAMIC_BLOCKS non-fixed blocks (5, so 9 blocks in total).
    Creating a block at the cap pages out the least recently accessed non-fixed
    block (last_accessed, falling back to last_modified). A paged-out block leaves
    the DLL but its row stays in L3 (LanceDB user_memory).

Divergences from the scaffold's stub, left open rather than resolved in the port:
  - the stub says a hard cap of 12 active blocks (config.MAX_ACTIVE_BLOCKS). 12 is
    the travel-agent APU's 4 fixed + 8 dynamic; Akili is 4 + 5. Not wired.
  - the stub says page-fault detection and page-in on later reference to a
    paged-out block. Akili does not implement it: a paged-out block can still be
    returned by the L3 search and reach the prompt as retrieved content, but it is
    never re-inserted into the DLL. Page-in exists only in the travel-agent APU
    (apu/mmu/controller.py, "SEMANTIC PAGE FAULT CHECK").
"""

import asyncio
import fcntl
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List

from apu import config
from apu.core.block_proposal import validate as validate_proposal
from apu.embeddings import local_embedder
from apu.logger import get_logger
from apu.mmu import cache_l1
from apu.mmu.block_types import refuse_non_tutoring_block_type
from apu.storage import lance_driver

logger = get_logger(__name__)

# ── Dynamic Isolation Locks ──────────────────────────────────────────────────
_dll_locks: Dict[str, asyncio.Lock] = {}

def get_dll_lock(agent_id: str) -> asyncio.Lock:
    """Get or create an asyncio.Lock for a specific agent."""
    if agent_id not in _dll_locks:
        _dll_locks[agent_id] = asyncio.Lock()
    return _dll_locks[agent_id]

# ── Adaptive certainty thresholds by block type (Education context) ───────────
# Defined in config because they are calibrated to the embedding model; see the
# measurements there. Re-exported here so call sites keep one import.
CERTAINTY_THRESHOLDS = config.CERTAINTY_THRESHOLDS

# Minimum certainty for a block to be included in the working context
MIN_RELEVANCE_CERTAINTY = config.MIN_RELEVANCE_CERTAINTY


# ═════════════════════════════════════════════════════════════════════════════
# DLL state (from Akili's mmu/controller.py)
# ═════════════════════════════════════════════════════════════════════════════

async def init_dll() -> dict:
    """
    Initialize the Living DLL with 4 fixed core blocks.
    Vectors are managed by LanceDB (local).
    """
    logger.debug("Initializing DLL — setting up fixed blocks...")

    dll = {
        "agent_id": f"agent-{uuid.uuid4()}",
        "head_id": "current_session",
        "tail_id": "student_profile",
        "dynamic_block_count": 0,
        "dynamic_block_max": config.MAX_DYNAMIC_BLOCKS,
        "created_at": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "course_selection": {
            "class": config.EDU_DEFAULT_CLASS,
            "subject": config.EDU_DEFAULT_SUBJECT
        },
        "nodes": {
            "current_session": {
                "id": "current_session",
                "label": "Current Session",
                "type": "temp",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["session", "current", "question", "explain", "today", "now", "help"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": None,
                "next": "active_course"
            },
            "active_course": {
                "id": "active_course",
                "label": "Active Course",
                "type": "cours",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["course", "subject", "chapter", "lesson", "exercise", "exam", "homework", "assignment", "topic"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": "current_session",
                "next": "learning_preferences"
            },
            "learning_preferences": {
                "id": "learning_preferences",
                "label": "Learning Preferences",
                "type": "fondamental",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["learning", "style", "difficulty", "strength", "weakness", "method", "preference", "practice"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": "active_course",
                "next": "student_profile"
            },
            "student_profile": {
                "id": "student_profile",
                "label": "Student Profile",
                "type": "fondamental",
                "is_fixed": True,
                "created_by": "system",
                "keywords": ["name", "age", "level", "grade", "school", "language", "goal", "background"],
                "active": True,
                "access_count": 0,
                "last_accessed": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "prev": "learning_preferences",
                "next": None
            }
        }
    }

    save_dll(dll)
    return dll


async def force_reinit_dll() -> dict:
    """
    Forcefully resets the DLL to its original 4-block state.
    Deletes all dynamic paging history.
    """
    logger.warning("FORCED REINIT: Wiping dynamic memory blocks...")
    return await init_dll()


async def switch_course(class_level: str, subject: str) -> dict:
    """
    Updates the active course context in the DLL without reinitializing memory.
    Updates course_selection and refreshes the active_course node label.
    """
    dll = await load_dll()
    dll["course_selection"] = {"class": class_level, "subject": subject}

    # Update the label of the active_course node to reflect the new context
    if "active_course" in dll["nodes"]:
        dll["nodes"]["active_course"]["label"] = f"{class_level.upper()} — {subject.title()}"
        dll["nodes"]["active_course"]["last_modified"] = datetime.now().isoformat()

    save_dll(dll)
    logger.info("Course switched to: %s/%s", class_level, subject)
    return dll


async def load_dll(agent_id: str | None = None) -> dict:
    """
    Load the DLL state from disk. Initializes a fresh DLL if no file exists.
    """
    path = config.METADATA_LINKS_PATH
    if not os.path.exists(path):
        return await init_dll()

    with open(path, encoding="utf-8") as f:
        dll = json.load(f)

    # Ensure course selection exists
    if not isinstance(dll.get("course_selection"), dict):
        dll["course_selection"] = {
            "class": config.EDU_DEFAULT_CLASS,
            "subject": config.EDU_DEFAULT_SUBJECT
        }
        save_dll(dll)

    return dll


def save_dll(dll: dict) -> None:
    """
    Persist the DLL state to disk (JSON): exclusive lock, then an atomic replace.

    The lock is taken on a separate lock file, and the temporary file carries the writer's
    pid. Locking the temporary file itself did not work: opening it with "w" truncates it
    BEFORE the lock is taken, so two writers sharing one temporary name could each wipe the
    other's half-written JSON.
    """
    dll["last_modified"] = datetime.now().isoformat()
    path = config.METADATA_LINKS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(path + ".lock", "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(dll, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def get_head_threshold(dll: dict) -> float:
    """Return the adaptive certainty threshold based on the HEAD node type."""
    head_node = dll["nodes"][dll["head_id"]]
    return CERTAINTY_THRESHOLDS.get(head_node["type"], 0.55)


async def search_memory(
    query_vector: List[float],
    class_level: str,
    subject: str,
    dll: dict | None = None,
) -> List[Dict]:
    """
    Bidirectional Metadata Jump (BMJ) — powered by LanceDB vector search.

    `dll` is the caller's DLL handle. When supplied, the BMJ promotion is applied
    to *that* object, so a caller holding a DLL across the turn sees the routing
    decision and any later save_dll of its own handle preserves it. When omitted
    (standalone use), a private copy is loaded.

    Without this, the planner's handle and this function's private copy were two
    different dicts, and the last writer (the memory write-back, holding the
    pre-promotion copy) silently reverted every routing decision the turn made.
    """
    logger.debug("DLL Search | class='%s' subject='%s'", class_level, subject)

    results = await lance_driver.search_block_index(
        query_vector,
        limit=12,
        class_level=class_level,
        subject=subject
    )

    filtered = []
    for res in results:
        b_type = res.get("block_type", "cours")
        threshold = CERTAINTY_THRESHOLDS.get(b_type, MIN_RELEVANCE_CERTAINTY)
        if res["certainty"] >= threshold:
            filtered.append(res)

    # BMJ Algorithm: Move the most relevant DLL memory node to HEAD.
    # Only applies to DLL nodes (student_profile, learning_preferences, etc.),
    # NOT to course content blocks returned from LanceDB (chapitre_X, etc.).
    if filtered:
        # Mutate the caller's handle when there is one, so the turn holds a
        # single DLL object and no later write can revert this promotion.
        working_dll = dll if dll is not None else await load_dll()
        dll_node_ids = set(working_dll.get("nodes", {}).keys())
        for block in filtered:
            block_id = block.get("block_id", "")
            if block_id in dll_node_ids:
                record_access(block_id, working_dll)
                move_to_front(block_id, working_dll)
                save_dll(working_dll)
                logger.debug("BMJ | Moved to HEAD: %s", block_id)
                break  # Only promote the first matching DLL node

    return filtered


def record_access(block_id: str, dll: dict) -> None:
    """
    Mark a block as accessed now.

    `last_accessed` and `access_count` used to be written at creation and updated
    by no code path, so "least recently accessed" eviction had nothing to sort on.
    Called from every read path: the L1 cache, retrieval, and content updates.
    """
    node = dll.get("nodes", {}).get(block_id)
    if node is None:
        return
    node["last_accessed"] = datetime.now().isoformat()
    node["access_count"] = int(node.get("access_count") or 0) + 1


def toggle_block(block_id: str, state: bool, dll: dict) -> dict:
    """Enable or disable a block."""
    if block_id in dll["nodes"]:
        dll["nodes"][block_id]["active"] = state
    return dll


async def update_node_content(block_id: str, content: str, dll: dict) -> dict:
    """
    Updates a node's content in both the DLL (metadata) and LanceDB (user_memory).
    Follows the invalidation pattern:
        1. Invalidate L1 (prevent stale reads during write window)
        2. Update LanceDB (L3 persistence)
        3. Re-populate L1 (write-back)
        4. Update DLL JSON (L2 index)
    """
    if block_id not in dll["nodes"]:
        return dll

    node = dll["nodes"][block_id]

    # 1. Invalidate L1 before write
    cache_l1.invalidate(block_id)

    # 2. Persist to LanceDB (L3) — user_memory table. The embedder is
    # process-cached, so this does not reload the model on every write-back.
    vector = await local_embedder.get_embedder().aembed_query(content)

    await lance_driver.upsert_local_block(
        block_id=block_id,
        content=content,
        block_type=node["type"],
        class_level=dll.get("course_selection", {}).get("class", "general"),
        subject=dll.get("course_selection", {}).get("subject", "general"),
        vector=vector
    )

    # 3. Write-back to L1 with correct TTL
    cache_l1.set(block_id, content, block_type=node["type"])

    # 4. Update DLL JSON metadata
    node["content"] = content
    node["last_modified"] = datetime.now().isoformat()
    record_access(block_id, dll)
    save_dll(dll)

    return dll


def move_to_front(block_id: str, dll: dict) -> dict:
    """
    Move the selected node to HEAD position (BMJ algorithm).

    An unknown id is a no-op, matching page_out_block. The two used to disagree:
    page_out_block returned silently while this raised KeyError, so the same stale
    id produced different outcomes depending on which one saw it first.
    """
    if block_id not in dll.get("nodes", {}):
        logger.debug("move_to_front: unknown block '%s' — ignoring", block_id)
        return dll
    if dll["head_id"] == block_id:
        return dll
    nodes = dll["nodes"]
    target = nodes[block_id]
    prev_id, next_id = target["prev"], target["next"]
    if prev_id: nodes[prev_id]["next"] = next_id
    if next_id: nodes[next_id]["prev"] = prev_id
    if dll["tail_id"] == block_id: dll["tail_id"] = prev_id
    old_head = dll["head_id"]
    nodes[old_head]["prev"] = block_id
    target["prev"], target["next"] = None, old_head
    dll["head_id"] = block_id
    return dll


def get_all_nodes(dll: dict) -> list:
    """Return all nodes in HEAD → TAIL order (DLL traversal)."""
    nodes, current = [], dll["head_id"]
    visited = set()
    while current and current not in visited:
        visited.add(current)
        nodes.append(dll["nodes"][current])
        current = dll["nodes"][current]["next"]
    return nodes


def _head_to_tail_order(dll: dict) -> list:
    """Traverse DLL from HEAD to TAIL, returning ordered node IDs."""
    order, current = [], dll["head_id"]
    visited = set()
    while current and current not in visited:
        visited.add(current)
        order.append(current)
        current = dll["nodes"][current]["next"]
    return order


def _tail_to_head_order(dll: dict) -> list:
    """Traverse DLL from TAIL to HEAD, returning ordered node IDs."""
    order, current = [], dll["tail_id"]
    visited = set()
    while current and current not in visited:
        visited.add(current)
        order.append(current)
        current = dll["nodes"][current]["prev"]
    return order


# ═════════════════════════════════════════════════════════════════════════════
# Block lifecycle (from Akili's mmu/block_factory.py)
# ═════════════════════════════════════════════════════════════════════════════

def insert_node_by_type(block_type: str, new_node: dict, dll: dict) -> dict:
    """
    Inserts a new node based on its semantic priority:
        temp        → HEAD (recent context)
        projet      → Middle (active planning)
        fondamental → Before TAIL (permanent knowledge)
    """
    refuse_non_tutoring_block_type(block_type)
    nodes = dll["nodes"]

    if block_type == "temp":
        old_head = dll["head_id"]
        new_node["next"] = old_head
        new_node["prev"] = None
        if old_head:
            nodes[old_head]["prev"] = new_node["id"]
        dll["head_id"] = new_node["id"]

    elif block_type == "fondamental":
        old_tail = dll["tail_id"]
        if old_tail:
            prev_to_tail = nodes[old_tail]["prev"]
            new_node["next"] = old_tail
            new_node["prev"] = prev_to_tail
            nodes[old_tail]["prev"] = new_node["id"]
            if prev_to_tail:
                nodes[prev_to_tail]["next"] = new_node["id"]
            else:
                dll["head_id"] = new_node["id"]
        else:
            dll["head_id"] = dll["tail_id"] = new_node["id"]

    else:  # projet — insert after HEAD
        head = dll["head_id"]
        if head:
            next_to_head = nodes[head]["next"]
            new_node["prev"] = head
            new_node["next"] = next_to_head
            nodes[head]["next"] = new_node["id"]
            if next_to_head:
                nodes[next_to_head]["prev"] = new_node["id"]
            else:
                dll["tail_id"] = new_node["id"]
        else:
            dll["head_id"] = dll["tail_id"] = new_node["id"]

    nodes[new_node["id"]] = new_node
    return dll

async def delete_block_stitching(block_id: str, dll: dict) -> dict:
    """
    Deletes a block from the DLL and local LanceDB.
    """
    agent_id = dll.get("agent_id")
    async with get_dll_lock(agent_id):
        nodes = dll["nodes"]
        if block_id not in nodes:
            raise ValueError(f"Block '{block_id}' does not exist.")

        target = nodes[block_id]
        if target.get("is_fixed", False):
            raise ValueError(f"Block '{block_id}' is fixed and cannot be deleted.")

        # 1. Local deletion in LanceDB.
        # NOTE: lance_driver has no delete_local_block, in Akili either. This call
        # raises AttributeError; ported as-is and reported in HACKATHON.md.
        await lance_driver.delete_local_block(block_id)

        # 2. Update DLL chain
        prev_id, next_id = target["prev"], target["next"]
        if prev_id:
            nodes[prev_id]["next"] = next_id
        if next_id:
            nodes[next_id]["prev"] = prev_id

        if dll["head_id"] == block_id:
            dll["head_id"] = next_id
        if dll["tail_id"] == block_id:
            dll["tail_id"] = prev_id

        del nodes[block_id]
        dll["dynamic_block_count"] = max(0, dll["dynamic_block_count"] - 1)

        save_dll(dll)
        logger.info(f"Block '{block_id}' deleted locally.")

    return dll

async def page_out_block(block_id: str, dll: dict) -> dict:
    """
    Deactivates a block (swap to disk). It stays in LanceDB but leaves the active DLL.
    """
    nodes = dll["nodes"]
    if block_id not in nodes:
        return dll

    target = nodes[block_id]
    if target.get("is_fixed", False):
        return dll

    prev_id, next_id = target["prev"], target["next"]
    if prev_id:
        nodes[prev_id]["next"] = next_id
    if next_id:
        nodes[next_id]["prev"] = prev_id

    if dll["head_id"] == block_id:
        dll["head_id"] = next_id
    if dll["tail_id"] == block_id:
        dll["tail_id"] = prev_id

    del nodes[block_id]
    dll["dynamic_block_count"] = max(0, dll["dynamic_block_count"] - 1)

    save_dll(dll)
    logger.info(f"Block '{block_id}' PAGED OUT (Moved to local storage).")
    return dll

def _refuse_zero_vector(block_id: str):
    raise ValueError(
        f"create_dynamic_block('{block_id}') was given no vector. Writing a "
        f"zero vector would put an unsearchable row into a semantic index; "
        f"embed the content first (see auto_execute_block_proposal)."
    )


async def create_dynamic_block(
    block_id: str,
    label: str,
    block_type: str,
    initial_content: str,
    keywords: list[str],
    created_by: str,
    dll: dict,
    vector: list[float] | None = None
) -> dict:
    """
    Creates a dynamic block in the DLL and LanceDB.
    """
    # Before the eviction below: refusing a forbidden block must not page out a real one.
    refuse_non_tutoring_block_type(block_type)

    if dll["dynamic_block_count"] >= dll["dynamic_block_max"]:
        # Semantic MMU: page out the least recently accessed block.
        #
        # `.get(key, default)` returned the STORED None rather than the default,
        # so min() compared None < None and raised TypeError: the cap raised
        # instead of evicting, and the working set was never bounded at all.
        # A node that has never been accessed sorts oldest, by creation time.
        dynamic_nodes = [n for n in dll["nodes"].values() if not n.get("is_fixed")]
        if dynamic_nodes:
            def _lru_key(node):
                return (
                    node.get("last_accessed")
                    or node.get("last_modified")
                    or "1970-01-01T00:00:00"
                )
            lru_node = min(dynamic_nodes, key=_lru_key)
            await page_out_block(lru_node["id"], dll)

    if block_id in dll["nodes"]:
        raise ValueError(f"Block '{block_id}' already exists.")

    # 1. Save in local LanceDB
    await lance_driver.upsert_local_block(
        block_id=block_id,
        content=initial_content,
        block_type=block_type,
        class_level="local",
        subject="local",
        # A zero vector is equidistant from every query, so the block would be
        # written and never retrievable. Callers must supply a real embedding.
        vector=vector if vector else _refuse_zero_vector(block_id),
    )

    # 2. Update Local DLL State
    new_node = {
        "id": block_id,
        "label": label,
        "type": block_type,
        "is_fixed": False,
        "created_by": created_by,
        "keywords": keywords,
        "active": True,
        "access_count": 0,
        "last_accessed": None,
        "last_modified": datetime.now().isoformat(),
        "prev": None,
        "next": None,
    }

    dll = insert_node_by_type(block_type, new_node, dll)
    dll["dynamic_block_count"] += 1
    save_dll(dll)

    return dll

async def update_block_content(
    block_id: str,
    new_content: str,
    new_keywords: list[str],
    dll: dict,
    vector: list[float] | None = None
) -> dict:
    """
    Updates a block's content locally.

    `new_keywords` and `vector` are accepted and ignored, as in Akili:
    update_node_content re-embeds the content itself and never touches keywords.
    """
    nodes = dll["nodes"]
    if block_id not in nodes:
        raise ValueError(f"Block '{block_id}' not found.")

    agent_id = dll.get("agent_id")

    async with get_dll_lock(agent_id):
        # Use centralized update_node_content to persist to both DLL and LanceDB
        dll = await update_node_content(block_id, new_content, dll)

    return dll

async def auto_execute_block_proposal(proposal: dict) -> bool:
    """
    Execute a block proposal, or refuse it.

    Returns True only if the block was really created and is really searchable.
    Callers must honour the return value: Akili's dashboard used to announce
    "Entry created!" regardless.
    """
    problem = validate_proposal(proposal)
    if problem:
        logger.error("Refusing block proposal: %s | %r", problem, proposal)
        return False

    try:
        # A real embedding of the real content. Without one, create_dynamic_block
        # used to fall back to a zero vector, equidistant from every query: the
        # block was written and could never be retrieved. Refusing beats writing a
        # permanently invisible block.
        vector = await local_embedder.get_embedder().aembed_query(proposal["initial_content"])
    except Exception as e:
        logger.error("Refusing block proposal: could not embed content (%s)", e)
        return False

    try:
        dll_latest = await load_dll()
        await create_dynamic_block(
            block_id=proposal["proposed_id"],
            label=proposal["label"],
            block_type=proposal["type"],
            initial_content=proposal["initial_content"],
            keywords=proposal.get("keywords", []),
            created_by="Akili",
            dll=dll_latest,
            vector=vector,
        )
        return True
    except Exception as e:
        logger.error(f"Auto-execute failed: {e}")
        return False
