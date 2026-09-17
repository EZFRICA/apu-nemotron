"""
Sync manager control flow. No GCS, no HTTP: the fetchers are stubbed.

Target: apu/sync/sync_manager.py (sync_with_registry, get_remote_catalog), ported from Akili
"""

import ast
import inspect
import json
import os
import textwrap

import pytest

from apu import config
from apu.sync import sync_manager


@pytest.fixture
def stub_fetch(monkeypatch):
    """Replace both fetchers; returns a dict you can mutate per test."""
    state = {"manifest": None, "blob": None}

    async def fake_fetch_json(url):
        return state["manifest"]

    async def fake_fetch_remote_json(blob_name):
        return state["blob"]

    monkeypatch.setattr(sync_manager, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(sync_manager, "_fetch_remote_json", fake_fetch_remote_json)
    return state


def _prompts_path():
    return os.path.join(os.path.dirname(config.LANCE_DB_PATH), "prompts.json")


async def test_unreachable_registry_reports_it(stub_fetch, no_network):
    stub_fetch["blob"] = None
    ok, msg = await sync_manager.sync_with_registry()
    assert ok is False
    assert "Unable to reach" in msg


async def test_a_manifest_without_prompts_is_not_an_error(stub_fetch, no_network):
    stub_fetch["blob"] = {"files": []}
    ok, msg = await sync_manager.sync_with_registry()
    assert ok is True
    assert "nothing to update" in msg


async def test_a_failed_prompt_download_is_reported(
    stub_fetch, no_network, akili_paths, monkeypatch
):
    calls = {"n": 0}

    async def fetch(blob_name):
        calls["n"] += 1
        return {"prompts": {"hash": "sha256:deadbeef"}} if calls["n"] == 1 else None

    monkeypatch.setattr(sync_manager, "_fetch_remote_json", fetch)
    ok, msg = await sync_manager.sync_with_registry()
    assert ok is False
    assert "Could not download" in msg


async def test_prompts_are_written_where_the_agent_reads_them(
    stub_fetch, no_network, akili_paths, monkeypatch
):
    """The agent loads prompts.json from next to the LanceDB directory."""
    payload = {"system_tutor": "You are Akili.", "6eme": "Be gentle."}
    requested = []

    async def fetch(blob_name):
        requested.append(blob_name)
        return {"prompts": {"hash": "sha256:deadbeef"}} if len(requested) == 1 else payload

    monkeypatch.setattr(sync_manager, "_fetch_remote_json", fetch)
    ok, msg = await sync_manager.sync_with_registry()
    assert ok is True, msg
    assert requested == ["manifest.json", "prompts/prompts_v1.json"]

    with open(_prompts_path()) as f:
        assert json.load(f) == payload


async def test_an_unchanged_hash_skips_the_download(
    stub_fetch, no_network, akili_paths
):
    path = _prompts_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"system_tutor": "x"}, f)

    stub_fetch["blob"] = {"prompts": {"hash": sync_manager.get_file_hash(path)}}

    ok, msg = await sync_manager.sync_with_registry()
    assert ok is True
    assert "up to date" in msg


async def test_the_remote_catalog_comes_from_the_manifest(stub_fetch, no_network):
    stub_fetch["blob"] = {"catalog": {"6eme": ["math", "history"]}}
    assert await sync_manager.get_remote_catalog() == {"6eme": ["math", "history"]}


async def test_an_unreachable_registry_yields_an_empty_catalog(stub_fetch, no_network):
    assert await sync_manager.get_remote_catalog() == {}


async def test_missing_gcs_credentials_make_the_registry_unreachable_not_a_crash(
    monkeypatch, no_network
):
    """Akili reads through an authenticated client; without credentials it reports."""
    async def no_client():
        return None

    monkeypatch.setattr(sync_manager, "_get_storage_client", no_client)
    assert await sync_manager.get_remote_catalog() == {}
    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok is False
    assert "Unable to reach the registry" in msg


def test_sync_with_registry_has_no_duplicate_of_download_prompts():
    tree = ast.parse(textwrap.dedent(inspect.getsource(sync_manager.sync_with_registry)))
    assigned = {
        t.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Name)
    }
    # The flag Akili read before assigning. Assigning it again would be the defect.
    assert "updated" not in assigned


def test_the_bucket_is_parsed_from_the_manifest_url(monkeypatch):
    """
    Carried over from Akili: there is no bucket setting on the device, the bucket is
    the 4th segment of the manifest URL.
    """
    assert sync_manager._registry_bucket_name() == "akili-registry"
    monkeypatch.setattr(
        config, "REGISTRY_MANIFEST_URL",
        "https://storage.googleapis.com/my-school-registry/manifest.json",
    )
    assert sync_manager._registry_bucket_name() == "my-school-registry"
    assert not hasattr(config, "GCS_BUCKET_NAME")
