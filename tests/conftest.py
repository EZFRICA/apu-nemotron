"""
Shared fixtures, ported from Akili's characterization suite.

Invariants this suite maintains:
  * No real API key is required and no INET socket is opened by the code under test.
  * The Nebius client is replaced by an in-process fake (`fake_nebius`), so every
    test that reaches inference sees exactly which model and parameters were used.
  * Every test gets its own LanceDB directory and its own metadata_links.json.
    The developer's data/ directory is never touched.
  * Vectors are small, hand-written and deterministic unless a test opts into the
    real ONNX embedder through `real_local_embedder`.
"""

import os
import socket
from types import SimpleNamespace

import pytest

# nebius_client builds its OpenAI client at import and raises without a key. A fake,
# non-empty key lets it import; the client itself is replaced by `fake_nebius`
# before any call, and sockets are blocked in the tests that reach inference, so the
# value never leaves the process.
os.environ["NEBIUS_API_KEY"] = "test-key-not-real-do-not-use"

from apu import config  # noqa: E402

# The real embedder identity, captured before any fixture swaps in the stub one.
_REAL_EMBEDDING_MODEL = config.LOCAL_EMBEDDING_MODEL
_REAL_EMBEDDING_DIM = config.LOCAL_EMBEDDING_DIM
_REAL_EMBEDDING_CACHE_DIR = config.LOCAL_EMBEDDING_CACHE_DIR


# ── hand-built deterministic vectors ─────────────────────────────────────────
DIM = 8


def vec(*first_values):
    """An 8-dim vector whose leading components are given, rest zero."""
    v = [0.0] * DIM
    for i, x in enumerate(first_values):
        v[i] = float(x)
    return v


V_A = vec(1.0)                      # unit, axis 0
V_B = vec(0.0, 1.0)                 # unit, axis 1 — orthogonal to V_A
V_A_SCALED = vec(3.0)               # same direction as V_A, 3x magnitude
V_A_OPPOSITE = vec(-1.0)            # antiparallel to V_A


# ── network lockdown ─────────────────────────────────────────────────────────
class NetworkBlocked(Exception):
    """Raised when code under test tries to open an INET socket."""


_real_socket = socket.socket


@pytest.fixture
def no_network(monkeypatch):
    """
    Block outbound network.

    Blocks AF_INET/AF_INET6 only, deliberately: `import lancedb` builds a
    background asyncio loop that needs socket.socketpair() -> AF_UNIX, so a
    blanket ban makes the package unimportable.
    """
    def guarded(family=socket.AF_INET, *a, **k):
        if family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(f"socket.socket(family={family!r})")
        return _real_socket(family, *a, **k)

    def blocked(*a, **k):
        raise NetworkBlocked("outbound connection attempt")

    monkeypatch.setattr(socket, "socket", guarded)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    return NetworkBlocked


# ── fake Nebius Token Factory client ─────────────────────────────────────────
class FakeNebiusClient:
    """
    Stands in for the OpenAI client inside apu.inference.nebius_client.

    Replaces the module's `_client` rather than call_main_model /
    call_extraction_model themselves, so nebius_client's own code still runs and
    the tests verify the real routing: which model id each call is sent to, and
    with which parameters.

    Replies are queued per model. The last reply of a queue is sticky, so a test
    can script a sequence or set one reply for every call. An Exception in the
    queue is raised instead of returned, to simulate a failing endpoint.
    """

    def __init__(self):
        self.main_replies = []
        self.extraction_replies = []
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if model == config.MAIN_MODEL:
            queue = self.main_replies
        elif model == config.EXTRACTION_MODEL:
            queue = self.extraction_replies
        else:
            raise AssertionError(f"unexpected model id sent to Nebius: {model!r}")

        reply = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else ""
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    @property
    def main_calls(self):
        return [c for c in self.calls if c["model"] == config.MAIN_MODEL]

    @property
    def extraction_calls(self):
        return [c for c in self.calls if c["model"] == config.EXTRACTION_MODEL]


@pytest.fixture
def fake_nebius(monkeypatch):
    from apu.inference import nebius_client

    fake = FakeNebiusClient()
    monkeypatch.setattr(nebius_client, "_client", fake)
    return fake


# ── isolated storage ─────────────────────────────────────────────────────────
@pytest.fixture
def akili_paths(tmp_path, monkeypatch):
    """
    Point config at a per-test temp dir and drop lance_driver's cached
    connection so the next get_db() re-opens against the temp path.
    """
    from apu.storage import lance_driver

    db_path = tmp_path / "akili_db"
    meta_path = tmp_path / "memory" / "metadata_links.json"

    monkeypatch.setattr(config, "LANCE_DB_PATH", str(db_path))
    monkeypatch.setattr(config, "METADATA_LINKS_PATH", str(meta_path))
    monkeypatch.setattr(config, "EMBEDDING_STAMP_PATH", str(tmp_path / "embedding_stamp.json"))
    # The suite's hand-built vectors are DIM-wide, so the configured embedder for
    # a test is a DIM-wide one. Without this, every storage test would trip the
    # dimension check against the real 384-dim default.
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_DIM", DIM)
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODEL", "test/stub-embedder")
    monkeypatch.setattr(lance_driver, "_db", None)

    yield {"db": str(db_path), "meta": str(meta_path), "root": tmp_path}

    lance_driver._db = None


@pytest.fixture(autouse=True)
def clean_l1():
    """L1 is a module-level singleton; wipe it between tests."""
    from apu.mmu import cache_l1
    cache_l1.flush_all()
    yield
    cache_l1.flush_all()


@pytest.fixture
def stub_embeddings(monkeypatch):
    """
    Replace the embedder with a deterministic stub returning V_A.

    Patches the CACHED INSTANCE, never the factory function: a module importing
    `get_embedder` by name while a patch is active would keep the stub after
    teardown. get_embedder() reads this global on every call, so patching it
    covers every call site however the name was imported.
    """
    from apu.embeddings import local_embedder

    calls = []

    class StubEmbedder:
        async def aembed_query(self, text):
            calls.append(("embed", text))
            return list(V_A)

        def embed_query(self, text):
            calls.append(("embed_sync", text))
            return list(V_A)

        def embed_documents(self, texts):
            calls.append(("embed_documents", list(texts)))
            return [list(V_A) for _ in texts]

    monkeypatch.setattr(local_embedder, "_embedder", StubEmbedder())
    return calls


@pytest.fixture
def real_local_embedder(monkeypatch):
    """
    Use the actual ONNX embedder, skipping if the model cache is absent.

    The suite must run on a machine that has not fetched the model, so any test
    needing real vectors opts in here rather than the whole suite depending on a
    240MB download. Restores the REAL model id and dimension over whatever
    `akili_paths` set.
    """
    from apu.embeddings import local_embedder

    monkeypatch.setattr(config, "LOCAL_EMBEDDING_MODEL", _REAL_EMBEDDING_MODEL)
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_DIM", _REAL_EMBEDDING_DIM)
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_CACHE_DIR", _REAL_EMBEDDING_CACHE_DIR)

    local_embedder.reset_embedder()
    try:
        embedder = local_embedder.get_embedder()
    except RuntimeError as e:
        pytest.skip(f"local embedding model not available: {str(e).splitlines()[0]}")
    yield embedder
    local_embedder.reset_embedder()


# ── DLL builders ─────────────────────────────────────────────────────────────
def make_node(node_id, node_type="temp", is_fixed=False, **extra):
    node = {
        "id": node_id,
        "label": node_id,
        "type": node_type,
        "is_fixed": is_fixed,
        "created_by": "test",
        "keywords": [],
        "active": True,
        "access_count": 0,
        "last_accessed": "2026-01-01T00:00:00",
        "last_modified": "2026-01-01T00:00:00",
        "prev": None,
        "next": None,
    }
    node.update(extra)
    return node


def make_chain(ids, types=None, fixed=()):
    """
    Build a well-formed DLL over `ids` in HEAD->TAIL order.
    `fixed` is the set of ids marked is_fixed.
    """
    types = types or {}
    nodes = {}
    for i, nid in enumerate(ids):
        nodes[nid] = make_node(
            nid,
            node_type=types.get(nid, "temp"),
            is_fixed=nid in fixed,
            prev=ids[i - 1] if i > 0 else None,
            next=ids[i + 1] if i < len(ids) - 1 else None,
        )
    return {
        "agent_id": "agent-test",
        "head_id": ids[0],
        "tail_id": ids[-1],
        "dynamic_block_count": sum(1 for n in ids if n not in fixed),
        "dynamic_block_max": 5,
        "course_selection": {"class": "6eme", "subject": "math"},
        "nodes": nodes,
    }


# ── invariant checker, shared by the DLL tests ───────────────────────────────
def check_dll_invariants(dll):
    """
    Returns a list of violated invariants (empty list == healthy chain).

    I1  HEAD->TAIL traversal terminates and reaches tail_id
    I2  TAIL->HEAD traversal terminates and reaches head_id
    I3  both traversals visit the same SET of ids
    I4  head_id has prev == None
    I5  tail_id has next == None
    I6  no node in `nodes` is orphaned (unreachable from HEAD)
    I7  every prev/next pointer targets an id that exists in `nodes`
    """
    problems = []
    nodes = dll["nodes"]
    head, tail = dll.get("head_id"), dll.get("tail_id")

    def walk(start, link):
        order, seen, cur = [], set(), start
        while cur is not None and cur not in seen:
            if cur not in nodes:
                order.append(cur)
                break
            seen.add(cur)
            order.append(cur)
            cur = nodes[cur].get(link)
        return order

    fwd = walk(head, "next")
    bwd = walk(tail, "prev")

    if not nodes:
        return problems

    if head is not None and fwd and fwd[-1] != tail:
        problems.append(f"I1 HEAD->TAIL ends at {fwd[-1]!r}, tail_id is {tail!r}")
    if tail is not None and bwd and bwd[-1] != head:
        problems.append(f"I2 TAIL->HEAD ends at {bwd[-1]!r}, head_id is {head!r}")
    if set(fwd) != set(bwd):
        problems.append(
            f"I3 forward set {sorted(set(fwd))} != backward set {sorted(set(bwd))}"
        )
    if head is not None and head in nodes and nodes[head].get("prev") is not None:
        problems.append(f"I4 head {head!r} has prev={nodes[head]['prev']!r}")
    if tail is not None and tail in nodes and nodes[tail].get("next") is not None:
        problems.append(f"I5 tail {tail!r} has next={nodes[tail]['next']!r}")
    orphans = set(nodes) - set(fwd)
    if orphans:
        problems.append(f"I6 orphaned nodes unreachable from HEAD: {sorted(orphans)}")
    for nid, n in nodes.items():
        for link in ("prev", "next"):
            tgt = n.get(link)
            if tgt is not None and tgt not in nodes:
                problems.append(f"I7 {nid!r}.{link} -> {tgt!r} which does not exist")

    return problems
