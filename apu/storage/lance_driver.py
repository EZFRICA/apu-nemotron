"""L3 storage: local LanceDB vector store (course registry + archived user memory).

Ported from Akili (app_local/storage/lance_driver.py). Behaviour unchanged; the
settings it reads now come from apu.config, and the embedder identity used for the
vector-space check is config.LOCAL_EMBEDDING_MODEL / LOCAL_EMBEDDING_DIM.
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

import lancedb

from apu import config
from apu.embeddings import local_embedder
from apu.mmu.block_types import TUTORING_EXCLUDED_BLOCK_TYPES, refuse_non_tutoring_block_type

_db = None
_db_lock = threading.Lock()

def get_db():
    """Initializes or retrieves the connection to the local LanceDB (Thread-safe)."""
    global _db
    with _db_lock:
        if _db is None:
            db_path = config.LANCE_DB_PATH
            os.makedirs(db_path, exist_ok=True)
            _db = lancedb.connect(db_path)
        return _db

def list_table_names(db=None) -> List[str]:
    """
    Return the table names in the local DB as a plain list of strings.

    lancedb changed this accessor's return type: up to ~0.29 `list_tables()`
    returned `list[str]`, on 0.30.x it returns a `ListTablesResponse` model whose
    names live on `.tables`. Membership tests against the response object are
    silently always False, which is how the entire L3 tier ended up inert in Akili.
    `table_names()` still works on 0.30.2 but is deprecated, so `.tables` is the
    stable read.

    This raises rather than degrading if the shape changes again. A silent empty
    list here disables course retrieval with no error anywhere, which is not
    something anyone will diagnose on a classroom machine.
    """
    db = get_db() if db is None else db
    raw = db.list_tables()

    if isinstance(raw, (list, tuple)):
        names = list(raw)
    else:
        tables = getattr(raw, "tables", None)
        if tables is None:
            raise TypeError(
                f"lancedb list_tables() returned {type(raw).__name__} with no "
                f"'.tables' attribute. The accessor's shape changed again; "
                f"apu.storage.lance_driver.list_table_names needs updating."
            )
        names = list(tables)

    bad = [n for n in names if not isinstance(n, str)]
    if bad:
        raise TypeError(
            f"lancedb list_tables() yielded non-string table names: {bad!r}. "
            f"apu.storage.lance_driver.list_table_names needs updating."
        )
    return names


# Re-exported from the embedder module so every writer and reader share ONE
# implementation: the certainty scale `1 - distance/2` is only cosine similarity if
# BOTH the stored vectors and the query are unit length.
normalize_vector = local_embedder.normalize_vector


def _sql_literal(value: str) -> str:
    """
    Quote a value for use in a LanceDB filter predicate.

    LanceDB's Python API exposes no parameter binding for `.where()`, so
    predicates have to be interpolated. Doubling any embedded single quote is the
    SQL-standard escape, so a block_id like "O'Brien" cannot terminate the literal
    early and rewrite the predicate. This matters most for the DELETE in
    upsert_local_block, where a malformed predicate is destructive rather than
    merely wrong.

    Scoped to this module's own id predicates. The class/subject filter in
    search_block_index is still interpolated unescaped, as in Akili.
    """
    return "'" + str(value).replace("'", "''") + "'"


class EmbeddingMismatch(RuntimeError):
    """The stored vectors were not produced by the configured embedder."""


def _read_stamp_doc() -> Dict:
    """The whole sidecar, keyed by table name. Empty dict if absent or corrupt."""
    path = config.EMBEDDING_STAMP_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        # A truncated sidecar must not break the read path on every turn.
        return {}
    if not isinstance(doc, dict):
        return {}
    # An older sidecar was a single flat {model, dim} for the whole store.
    # Treat it as applying to every table rather than discarding it.
    if "model" in doc:
        return {name: doc for name in ("edu_registry", "user_memory")}
    return doc


def read_stamp(table_name: str) -> Optional[Dict]:
    """
    Which embedder wrote this table, or None if never recorded.

    Stamped PER TABLE: `edu_registry` comes from the cloud registry and
    `user_memory` is written on-device, so they can legitimately be in different
    vector spaces, e.g. after re-embedding local memory without re-downloading
    courses. A single store-wide stamp reported the configured model for both
    and hid exactly that case.
    """
    return _read_stamp_doc().get(table_name)


def write_stamp(table_name: str) -> None:
    """Record the configured embedder as the owner of one table."""
    path = config.EMBEDDING_STAMP_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = _read_stamp_doc()
    doc[table_name] = {
        "model": config.LOCAL_EMBEDDING_MODEL,
        "dim": config.LOCAL_EMBEDDING_DIM,
        "updated_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def _table_vector_dim(table) -> Optional[int]:
    """Vector width from the Arrow schema, without reading any rows."""
    try:
        field = table.schema.field("vector")
    except KeyError:
        return None
    return getattr(field.type, "list_size", None) or None


def verify_embedding_space(table, table_name: str) -> None:
    """
    Refuse to search a table whose vectors are not in the configured space.

    Two independent checks, because neither alone is sufficient:

      - dimension, read from the Arrow schema. Always available, catches the
        common case (a registry published at 3072 against a client at 384).
      - model id, read from the local sidecar. Needed because two different
        models can share a dimension, which produces results that look
        plausible and are noise.

    Raising is the point. A vector-space mismatch does not fail on its own: it
    returns confidently ranked nonsense, which is indistinguishable from the
    system merely being bad at its job.
    """
    configured_dim = config.LOCAL_EMBEDDING_DIM
    stored_dim = _table_vector_dim(table)

    if stored_dim is not None and stored_dim != configured_dim:
        raise EmbeddingMismatch(
            f"Table '{table_name}' holds {stored_dim}-dimension vectors but the "
            f"configured embedder '{config.LOCAL_EMBEDDING_MODEL}' produces "
            f"{configured_dim}.\n"
            f"These are different vector spaces; searching would return noise "
            f"ranked as though it were relevant.\n"
            f"Fix: regenerate and re-download the course registry, and re-embed "
            f"local memory (Akili's scripts/migrate_embeddings.py, not yet ported)."
        )

    stamp = read_stamp(table_name)
    if stamp and stamp.get("model") and stamp["model"] != config.LOCAL_EMBEDDING_MODEL:
        raise EmbeddingMismatch(
            f"Table '{table_name}' was written by embedding model "
            f"'{stamp['model']}' but the configured model is "
            f"'{config.LOCAL_EMBEDDING_MODEL}'.\n"
            f"Same dimension does not mean the same vector space.\n"
            f"Fix: re-embed local memory (Akili's scripts/migrate_embeddings.py, "
            f"not yet ported) or set LOCAL_EMBEDDING_MODEL back to '{stamp['model']}'."
        )


def reset_local_db():
    """Wipes the local database entirely. Use with caution."""
    global _db
    import shutil
    with _db_lock:
        _db = None # Drop connection
        db_path = config.LANCE_DB_PATH
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"LanceDB at {db_path} has been wiped.")

async def search_block_index(query_vector: List[float], limit: int = 12,
                         class_level: str = None, subject: str = None) -> List[Dict]:
    """
    Unified semantic search in LanceDB across multiple tables (Courses + Memory).
    """
    db = get_db()
    all_results = []

    # Normalised on both sides so `1 - distance/2` is cosine similarity by
    # construction, whatever the embedder returns.
    query_vector = normalize_vector(query_vector)

    # Tables to search
    tables_to_search = ["edu_registry", "user_memory"]
    existing = list_table_names(db)

    for table_name in tables_to_search:
        if table_name not in existing:
            continue

        table = db.open_table(table_name)
        verify_embedding_space(table, table_name)

        # Build filter (optional for user_memory, strict for edu_registry)
        filter_query = ""
        if table_name == "edu_registry" and class_level and subject:
            filter_query = f"class_level = '{class_level}' AND subject = '{subject}'"

        # Search
        query = table.search(query_vector, vector_column_name="vector").limit(limit)
        if filter_query:
            query = query.where(filter_query)

        df = query.to_pandas()

        for _, row in df.iterrows():
            # Defence in depth: upsert_local_block already refuses these types, but a row
            # written some other way must still never reach a tutoring prompt.
            if row.get("block_type") in TUTORING_EXCLUDED_BLOCK_TYPES:
                continue
            dist = row.get("_distance", 0)
            certainty = 1 - (dist / 2)

            all_results.append({
                "block_id": row["id"],
                "chapter_id": row.get("chapter", row.get("id")), # Fallback to ID for memory
                "block_type": row.get("block_type", "memory" if table_name == "user_memory" else "cours"),
                "certainty": certainty,
                "content": row.get("content", ""),
                "source_table": table_name
            })

    # Sort all merged results by certainty
    all_results.sort(key=lambda x: x["certainty"], reverse=True)
    return all_results[:limit]

async def get_block_content(block_id: str) -> Optional[str]:
    """Retrieves the content of a DLL memory node from 'user_memory' table."""
    db = get_db()
    if "user_memory" not in list_table_names(db):
        return None

    table = db.open_table("user_memory")
    result = table.search().where(f"id = {_sql_literal(block_id)}").to_pandas()

    if result.empty:
        return None

    # upsert_local_block keeps one row per id, but a store written before that fix
    # can still hold several revisions. Order by updated_at so the newest wins
    # either way, rather than taking whichever row the scan yielded first.
    if "updated_at" in result.columns and len(result) > 1:
        result = result.sort_values("updated_at", ascending=False, kind="stable")
    return result.iloc[0]["content"]

async def upsert_local_block(block_id: str, content: str, block_type: str,
                           class_level: str, subject: str, vector: List[float]):
    """
    Allows the student to add their own blocks (notes, session)
    into a separate local table 'user_memory'.
    """
    refuse_non_tutoring_block_type(block_type)
    db = get_db()
    data = [{
        "id": block_id,
        "content": content,
        "block_type": block_type,
        "class_level": class_level,
        "subject": subject,
        "vector": normalize_vector(vector),
        "updated_at": datetime.now().isoformat()
    }]

    if "user_memory" not in list_table_names(db):
        try:
            db.create_table("user_memory", data=data)
            write_stamp("user_memory")
            return
        except Exception:
            # Another thread created it between the check and the call.
            pass

    # Record which embedder owns this table, so a later model swap is detected
    # rather than silently querying one vector space with another's vectors.
    write_stamp("user_memory")

    table = db.open_table("user_memory")
    # Real upsert: drop any existing revision of this id before adding the new
    # one. Append-only writes grew the table by one row (and one data fragment,
    # transaction and manifest version) per student turn, forever: unbounded
    # disk growth proportional to conversation length on hardware chosen for
    # having very little of it.
    table.delete(f"id = {_sql_literal(block_id)}")
    table.add(data)
