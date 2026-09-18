"""
Storage layer: table discovery, distance metric, certainty scale, upsert semantics.

Target: apu/storage/lance_driver.py (ported from Akili)
Verified against the installed lancedb 0.30.2 — nothing here is assumed from docs.
"""

import pytest

from apu.storage import lance_driver
from tests.conftest import DIM, V_A, V_A_OPPOSITE, V_A_SCALED, V_B


def _row(rid, vector, content="c", block_type="cours", cls="6eme", subject="math"):
    return {
        "id": rid,
        "chapter": rid,
        "content": content,
        "block_type": block_type,
        "class_level": cls,
        "subject": subject,
        "vector": list(vector),
        "updated_at": "2026-01-01T00:00:00",
    }


# ── what lancedb 0.30.2 actually returns ─────────────────────────────────────

def test_list_tables_does_not_return_a_list_of_strings(akili_paths):
    """
    Pins the root cause behind most of this file's defects. On lancedb 0.30.2
    list_tables() returns a ListTablesResponse model, and `"name" in response`
    is False even when the table exists.
    """
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A)])

    resp = db.list_tables()
    assert not isinstance(resp, list)
    assert type(resp).__name__ == "ListTablesResponse"
    assert "edu_registry" not in resp          # <- the guard that was used


def test_list_table_names_returns_plain_strings(akili_paths):
    """The helper every call site now routes through."""
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A)])
    db.create_table("user_memory", data=[_row("student_profile", V_A)])

    names = lance_driver.list_table_names(db)
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert sorted(names) == ["edu_registry", "user_memory"]
    assert "edu_registry" in names


def test_list_table_names_defaults_to_the_module_connection(akili_paths):
    lance_driver.get_db().create_table("user_memory", data=[_row("a", V_A)])
    assert lance_driver.list_table_names() == ["user_memory"]


def test_list_table_names_accepts_a_plain_list(akili_paths):
    """Older lancedb returned list[str] directly; that shape must keep working."""
    class _OldDB:
        def list_tables(self):
            return ["edu_registry", "user_memory"]

    assert lance_driver.list_table_names(_OldDB()) == ["edu_registry", "user_memory"]


def test_list_table_names_raises_loudly_if_the_shape_changes_again(akili_paths):
    """
    The whole point of the helper. A silent [] here disables course retrieval
    with no error anywhere in the system, which is undiagnosable in the field.
    """
    class _FutureDB:
        def list_tables(self):
            return object()          # neither a sequence nor has .tables

    with pytest.raises(TypeError, match="no '.tables' attribute"):
        lance_driver.list_table_names(_FutureDB())


def test_list_table_names_raises_on_non_string_names(akili_paths):
    class _WeirdDB:
        def list_tables(self):
            return [("tables", ["a"]), ("page_token", None)]

    with pytest.raises(TypeError, match="non-string table names"):
        lance_driver.list_table_names(_WeirdDB())


def test_lancedb_default_metric_is_squared_l2_not_cosine(akili_paths):
    """
    No metric is set anywhere in lance_driver.py, so the table default applies.
    Measured, not assumed: identical=0, orthogonal=2, antiparallel=4,
    same-direction-3x=4. That is ||a-b||^2 (L2 squared), not cosine.
    Cosine would give 0 / 1 / 2 / 0.
    """
    db = lance_driver.get_db()
    t = db.create_table("edu_registry", data=[
        _row("same", V_A),
        _row("orth", V_B),
        _row("opposite", V_A_OPPOSITE),
        _row("scaled", V_A_SCALED),
    ])
    df = t.search(list(V_A), vector_column_name="vector").limit(10).to_pandas()
    dist = dict(zip(df["id"], df["_distance"], strict=True))

    assert dist["same"] == pytest.approx(0.0)
    assert dist["orth"] == pytest.approx(2.0)       # cosine would be 1.0
    assert dist["opposite"] == pytest.approx(4.0)   # cosine would be 2.0
    assert dist["scaled"] == pytest.approx(4.0)     # cosine would be 0.0


def test_certainty_formula_on_the_actual_metric(akili_paths):
    """
    certainty = 1 - dist/2  (lance_driver.py:70) applied to squared-L2 distances.
    Records the actual numbers the thresholds get compared against.
    """
    def certainty(d):
        return 1 - (d / 2)

    assert certainty(0.0) == pytest.approx(1.0)    # identical vector
    assert certainty(2.0) == pytest.approx(0.0)    # orthogonal
    assert certainty(4.0) == pytest.approx(-1.0)   # antiparallel
    # A vector pointing the SAME way but 3x longer also scores -1.0:
    assert certainty(4.0) == pytest.approx(-1.0)

    # For UNIT vectors, squared-L2 = 2 - 2*cos, so 1 - d/2 == cosine similarity
    # exactly. The formula is only correct if embeddings are L2-normalised.
    import math
    for angle_deg in (0, 45, 60, 90, 180):
        cos = math.cos(math.radians(angle_deg))
        sq_l2 = 2 - 2 * cos
        assert certainty(sq_l2) == pytest.approx(cos, abs=1e-9)


# ── search_block_index ───────────────────────────────────────────────────────

async def test_search_block_index_searches_both_tables(akili_paths):
    """
    Was `test_search_block_index_returns_nothing_even_with_an_exact_match`, which
    pinned the S5-A defect (search returned [] unconditionally). Now asserts the
    fixed behaviour: both tables are consulted and both exact matches come back.
    """
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A, content="Fractions.")])
    db.create_table("user_memory", data=[_row("student_profile", V_A, content="Marc")])

    results = await lance_driver.search_block_index(
        list(V_A), limit=12, class_level="6eme", subject="math"
    )
    by_id = {r["block_id"]: r for r in results}
    assert set(by_id) == {"ch1", "student_profile"}
    assert by_id["ch1"]["source_table"] == "edu_registry"
    assert by_id["student_profile"]["source_table"] == "user_memory"
    assert by_id["ch1"]["certainty"] == pytest.approx(1.0)


async def test_search_block_index_applies_the_class_subject_filter(akili_paths):
    """The filter is strict for edu_registry and absent for user_memory."""
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[
        _row("ch_match", V_A, cls="6eme", subject="math"),
        _row("ch_other", V_A, cls="5eme", subject="history"),
    ])

    results = await lance_driver.search_block_index(
        list(V_A), limit=12, class_level="6eme", subject="math"
    )
    assert [r["block_id"] for r in results] == ["ch_match"]


async def test_search_block_index_on_an_empty_db_returns_empty(akili_paths):
    assert await lance_driver.search_block_index(list(V_A), 12, "6eme", "math") == []


async def test_search_block_index_finds_an_exact_match(akili_paths):
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A, content="Fractions.")])

    results = await lance_driver.search_block_index(
        list(V_A), limit=12, class_level="6eme", subject="math"
    )
    assert len(results) == 1
    assert results[0]["block_id"] == "ch1"
    assert results[0]["certainty"] == pytest.approx(1.0)


async def test_orthogonal_query_returns_a_low_certainty_row(akili_paths):
    db = lance_driver.get_db()
    db.create_table("edu_registry", data=[_row("ch1", V_A, content="Fractions.")])

    results = await lance_driver.search_block_index(
        list(V_B), limit=12, class_level="6eme", subject="math"
    )
    assert len(results) == 1
    assert results[0]["certainty"] == pytest.approx(0.0)


# ── get_block_content ────────────────────────────────────────────────────────

async def test_get_block_content_returns_none_for_an_absent_id(akili_paths):
    """
    Was `test_get_block_content_returns_none_even_when_the_row_exists`, which
    pinned the S5-A defect. None is now reserved for its real meaning: no row.
    """
    db = lance_driver.get_db()
    db.create_table("user_memory", data=[_row("student_profile", V_A, content="Marc")])
    assert await lance_driver.get_block_content("no_such_block") is None


async def test_get_block_content_returns_none_when_the_table_is_missing(akili_paths):
    lance_driver.get_db()
    assert await lance_driver.get_block_content("student_profile") is None


async def test_get_block_content_returns_the_stored_content(akili_paths):
    db = lance_driver.get_db()
    db.create_table("user_memory", data=[_row("student_profile", V_A, content="Marc")])
    assert await lance_driver.get_block_content("student_profile") == "Marc"


# ── upsert semantics ─────────────────────────────────────────────────────────

async def _write(bid, content, vector=V_A):
    await lance_driver.upsert_local_block(
        block_id=bid, content=content, block_type="fondamental",
        class_level="6eme", subject="math", vector=list(vector),
    )


async def test_superseded_revisions_are_removed_from_disk(akili_paths):
    """
    Was `test_three_writes_to_one_id_produce_three_rows`, which asserted the
    append-only behaviour of C2. The distinct property here (as against the
    acceptance test, which checks what the *read* returns) is that the old
    revisions are actually gone from the table rather than merely shadowed.
    """
    await _write("student_profile", "revision one")
    await _write("student_profile", "revision two")
    await _write("student_profile", "revision three")

    db = lance_driver.get_db()
    contents = list(db.open_table("user_memory").to_pandas()["content"])
    assert contents == ["revision three"]
    assert "revision one" not in contents
    assert "revision two" not in contents


async def test_row_count_stays_flat_across_writes_to_one_id(akili_paths):
    """
    Was `test_row_count_grows_linearly_with_writes`, which asserted 10 writes ->
    10 rows. That unbounded growth is the C2 defect, not the contract. Storage
    is now proportional to the number of distinct blocks, not to conversation
    length -- the property the target hardware actually needs.
    """
    for i in range(10):
        await _write("current_session", f"turn {i}")
    db = lance_driver.get_db()
    df = db.open_table("user_memory").to_pandas()
    assert df.shape[0] == 1
    assert df.iloc[0]["content"] == "turn 9"


async def test_distinct_ids_still_get_their_own_rows(akili_paths):
    """The upsert must key on id only -- it must not collapse different blocks."""
    await _write("student_profile", "Marc, 12")
    await _write("current_session", "fractions")
    await _write("learning_preferences", "visual")
    await _write("student_profile", "Marc, 13")

    db = lance_driver.get_db()
    df = db.open_table("user_memory").to_pandas()
    assert sorted(df["id"]) == ["current_session", "learning_preferences", "student_profile"]
    assert await lance_driver.get_block_content("student_profile") == "Marc, 13"
    assert await lance_driver.get_block_content("current_session") == "fractions"


async def test_a_block_id_containing_a_quote_is_handled(akili_paths):
    """
    The upsert's DELETE predicate is interpolated. An unescaped single quote
    would either raise or, worse, change which rows the delete matches.
    """
    await _write("O'Brien", "first")
    await _write("O'Brien", "second")
    await _write("normal", "untouched")

    db = lance_driver.get_db()
    df = db.open_table("user_memory").to_pandas()
    assert sorted(df["id"]) == ["O'Brien", "normal"]
    assert await lance_driver.get_block_content("O'Brien") == "second"
    assert await lance_driver.get_block_content("normal") == "untouched"


async def test_get_block_content_after_three_writes_returns_some_revision(akili_paths):
    """
    Was `..._is_still_none`: the .iloc[0] arbitrary-revision problem used to be
    masked by S5-A, because get_block_content bailed out before selecting a row.
    With the accessor fixed it selects one, so the masking is gone. Which
    revision it picks is C2's job (test_upsert_keeps_one_row_and_returns_the_latest).
    """
    await _write("student_profile", "revision one")
    await _write("student_profile", "revision three")
    got = await lance_driver.get_block_content("student_profile")
    assert got in {"revision one", "revision three"}


async def test_upsert_keeps_one_row_and_returns_the_latest(akili_paths):
    await _write("student_profile", "revision one")
    await _write("student_profile", "revision two")
    await _write("student_profile", "revision three")

    db = lance_driver.get_db()
    df = db.open_table("user_memory").to_pandas()
    assert df.shape[0] == 1
    assert await lance_driver.get_block_content("student_profile") == "revision three"


# ── normalisation (decision O1) ──────────────────────────────────────────────

def test_normalize_vector_produces_unit_length():
    for v in (V_A, V_B, V_A_SCALED, V_A_OPPOSITE, [3.0, 4.0] + [0.0] * (DIM - 2)):
        out = lance_driver.normalize_vector(list(v))
        assert sum(x * x for x in out) == pytest.approx(1.0)
        assert len(out) == DIM


def test_normalize_vector_preserves_direction():
    out = lance_driver.normalize_vector(list(V_A_SCALED))
    assert out == pytest.approx(list(V_A))


def test_normalize_vector_leaves_a_zero_vector_alone():
    """A zero vector has no direction; it must not raise or produce NaNs."""
    out = lance_driver.normalize_vector([0.0] * DIM)
    assert out == [0.0] * DIM


async def test_a_scaled_copy_of_a_vector_scores_the_same_certainty(akili_paths):
    """
    Decision O1. Magnitude must not affect ranking: a query 3x longer than the
    stored vector is semantically identical and must score identically.

    Before normalisation this returned certainty -1.0 for the scaled query
    (squared-L2 distance 4.0), ranking an identical match below an orthogonal
    one and putting it below every CERTAINTY_THRESHOLD.
    """
    await _write("student_profile", "Marc", vector=list(V_A))

    unit = await lance_driver.search_block_index(list(V_A), limit=5)
    scaled = await lance_driver.search_block_index(list(V_A_SCALED), limit=5)

    assert [r["block_id"] for r in unit] == ["student_profile"]
    assert [r["block_id"] for r in scaled] == ["student_profile"]
    assert unit[0]["certainty"] == pytest.approx(1.0)
    assert scaled[0]["certainty"] == pytest.approx(unit[0]["certainty"])


async def test_a_scaled_stored_vector_scores_the_same_certainty(akili_paths):
    """The insert side of the same property."""
    await _write("plain", "a", vector=list(V_A))
    await _write("scaled", "b", vector=list(V_A_SCALED))

    results = await lance_driver.search_block_index(list(V_A), limit=5)
    by_id = {r["block_id"]: r["certainty"] for r in results}
    assert by_id["plain"] == pytest.approx(1.0)
    assert by_id["scaled"] == pytest.approx(by_id["plain"])


async def test_normalisation_keeps_the_orthogonal_and_opposite_ends_of_the_scale(
    akili_paths,
):
    """The scale must still span 1.0 / 0.0 / -1.0, so thresholds mean something."""
    await _write("same", "a", vector=list(V_A))
    await _write("orth", "b", vector=list(V_B))
    await _write("opposite", "c", vector=list(V_A_OPPOSITE))

    results = await lance_driver.search_block_index(list(V_A_SCALED), limit=5)
    by_id = {r["block_id"]: r["certainty"] for r in results}
    assert by_id["same"] == pytest.approx(1.0)
    assert by_id["orth"] == pytest.approx(0.0)
    assert by_id["opposite"] == pytest.approx(-1.0)


# ── predicate construction ───────────────────────────────────────────────────

async def test_filter_predicate_is_built_by_string_interpolation(akili_paths):
    """
    lance_driver.py builds SQL predicates with f-strings, since LanceDB exposes no
    parameter binding. A value containing a single quote produces a malformed predicate.
    class_level and subject come from the interface's course selection, bounded by the
    catalog -- but block_id is not, so id predicates escape their literal.
    """
    db = lance_driver.get_db()
    db.create_table("user_memory", data=[_row("normal", V_A)])
    table = db.open_table("user_memory")

    # lancedb surfaces the tokenizer failure as RuntimeError; the point is that an
    # unescaped quote terminates the literal and the predicate is rejected, not accepted.
    with pytest.raises(RuntimeError, match="Unterminated string literal"):
        table.search().where("id = 'O'Brien'").limit(1).to_pandas()
