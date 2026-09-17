"""
The download path: table detection, the embedding stamp, the local manifest.

Target: apu/sync/sync_manager.py (download_course), ported from Akili
Network is never touched; the GCS client and manifest fetch are stubbed.
"""

import ast
import pathlib

from apu import config
from apu.storage import lance_driver
from apu.sync import sync_manager
from tests.conftest import DIM
from tests.registry_fakes import course_rows, install_fake_registry, manifest


async def test_a_download_stamps_the_registry(akili_paths, monkeypatch):
    install_fake_registry(monkeypatch, akili_paths, manifest())
    assert lance_driver.read_stamp("edu_registry") is None

    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok, msg

    stamp = lance_driver.read_stamp("edu_registry")
    assert stamp["model"] == "test/stub-embedder"
    assert stamp["dim"] == DIM


async def test_parquets_are_read_from_the_bucket_root(akili_paths, monkeypatch):
    """Matches where batch_pipeline uploads them, not the manifest's /courses/ url."""
    requested = install_fake_registry(monkeypatch, akili_paths, manifest())
    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok, msg
    assert requested == ["manifest.json", "6eme_math_v1.parquet"]


async def test_a_downloaded_course_is_recorded_and_searchable(akili_paths, monkeypatch):
    install_fake_registry(monkeypatch, akili_paths, manifest())
    assert not sync_manager.is_course_available_locally("6eme", "math")

    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok, msg

    assert sync_manager.is_course_available_locally("6eme", "math")
    results = await lance_driver.search_block_index(
        course_rows()[0]["vector"], class_level="6eme", subject="math"
    )
    assert [r["block_id"] for r in results] == ["ch1"]
    assert not list(pathlib.Path(config.CACHE_DIR).glob("*.parquet")), "temp file left behind"


async def test_a_registry_built_with_another_model_is_refused_before_download(
    akili_paths, monkeypatch
):
    requested = install_fake_registry(
        monkeypatch, akili_paths, manifest(model="legacy/remote-embedder", dim=3072)
    )

    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok is False
    assert "legacy/remote-embedder" in msg
    assert "3072" in msg
    assert "test/stub-embedder" in msg
    assert requested == ["manifest.json"], "nothing was downloaded"
    assert "edu_registry" not in lance_driver.list_table_names()


async def test_an_unstamped_manifest_still_imports(akili_paths, monkeypatch, capsys):
    """A registry predating stamping must not become undownloadable."""
    install_fake_registry(monkeypatch, akili_paths, manifest(model=None))

    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok, msg
    assert "no embedding stamp" in capsys.readouterr().out


async def test_a_course_missing_from_the_registry_is_reported(akili_paths, monkeypatch):
    install_fake_registry(monkeypatch, akili_paths, manifest())
    ok, msg = await sync_manager.download_course("5eme", "history")
    assert ok is False
    assert "not found in the registry" in msg


async def test_a_second_download_replaces_rather_than_duplicating(
    akili_paths, monkeypatch
):
    install_fake_registry(monkeypatch, akili_paths, manifest())
    assert (await sync_manager.download_course("6eme", "math"))[0]
    assert (await sync_manager.download_course("6eme", "math"))[0]

    df = lance_driver.get_db().open_table("edu_registry").to_pandas()
    assert len(df) == 1, f"duplicated rows: {len(df)}"


async def test_downloading_one_course_keeps_the_others(akili_paths, monkeypatch):
    install_fake_registry(monkeypatch, akili_paths, manifest(course="6eme_math"),
                          rows=course_rows("6eme", "math"))
    assert (await sync_manager.download_course("6eme", "math"))[0]

    install_fake_registry(monkeypatch, akili_paths, manifest(course="5eme_history"),
                          rows=course_rows("5eme", "history"))
    assert (await sync_manager.download_course("5eme", "history"))[0]

    df = lance_driver.get_db().open_table("edu_registry").to_pandas()
    assert sorted(zip(df["class_level"], df["subject"])) == [
        ("5eme", "history"), ("6eme", "math"),
    ]


def test_sync_manager_does_not_call_list_tables_directly():
    """db.list_tables() returns a response model whose __contains__ never matches."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "apu" / "sync" / "sync_manager.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "list_tables", (
                "use lance_driver.list_table_names(); db.list_tables() returns a "
                "response model whose __contains__ never matches"
            )
