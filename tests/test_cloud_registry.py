"""
The publishing side of the cloud registry, and its contract with the device.

Target: cloud_registry/pipeline/batch_pipeline.py, cloud_registry/config/settings.py

Akili had no tests for the pipeline beyond the shared-embedding-config invariant.
These run it against a temporary registry directory with a stub embedder, then feed
its output to the real device-side download, so the two ends are checked together.
"""

import json
import math
import pathlib
import re

import pandas as pd
import pytest
import yaml

from apu import config
from apu.storage import lance_driver
from apu.sync import sync_manager
from tests.conftest import DIM, V_A_SCALED
from tests.registry_fakes import install_fake_registry

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY_PKG = REPO_ROOT / "cloud_registry"


@pytest.fixture
def pipeline(akili_paths, monkeypatch, tmp_path):
    """batch_pipeline pointed at a temp registry + course tree, with a stub embedder."""
    from cloud_registry.pipeline import batch_pipeline

    courses = tmp_path / "courses"
    (courses / "6eme" / "math").mkdir(parents=True)
    (courses / "6eme" / "math" / "index.md").write_text("# index, skipped")
    (courses / "6eme" / "math" / "chapter_1_simple_fractions.md").write_text(
        "# Fractions\nA fraction is a part of a whole."
    )
    (courses / "prompts").mkdir()
    (courses / "prompts" / "system_tutor.txt").write_text("You are Akili.")

    registry = tmp_path / "registry"
    registry.mkdir()

    class StubEmbedder:
        async def aembed_query(self, text):
            # Deliberately not unit length: the pipeline must normalise.
            return list(V_A_SCALED)

    monkeypatch.setattr(batch_pipeline, "COURSES_DIR", courses)
    monkeypatch.setattr(batch_pipeline, "REGISTRY_DIR", registry)
    monkeypatch.setattr(batch_pipeline, "MANIFEST_PATH", registry / "manifest.json")
    monkeypatch.setattr(batch_pipeline, "_embedder", StubEmbedder())
    return batch_pipeline


# ── the shared embedding config ──────────────────────────────────────────────

def test_the_registry_settings_do_not_declare_an_embedding_model():
    """How Akili's registry once shipped at 3072 dims while devices queried at 384."""
    text = (REGISTRY_PKG / "config" / "settings.py").read_text()
    assert re.search(r"^\s*\w*EMBEDDING\w*\s*=", text, re.M) is None


def test_the_pipeline_stamps_with_the_devices_embedding_config():
    from apu.embeddings.local_embedder import embedding_stamp
    assert embedding_stamp() == {
        "model": config.LOCAL_EMBEDDING_MODEL, "dim": config.LOCAL_EMBEDDING_DIM,
    }


def test_importing_the_pipeline_does_not_build_the_embedder():
    """Built with allow_download=True at import, it would download ~240MB."""
    from cloud_registry.pipeline import batch_pipeline
    assert batch_pipeline._embedder is None


# ── building the registry ────────────────────────────────────────────────────

async def test_a_course_folder_becomes_a_normalised_parquet(pipeline):
    assert await pipeline.process_one("6eme", "math") == 1

    df = pd.read_parquet(pipeline.REGISTRY_DIR / "6eme_math_v1.parquet")
    [row] = df.to_dict("records")
    assert row["id"] == "chapter_1_simple_fractions"
    assert row["class_level"] == "6eme" and row["subject"] == "math"
    assert row["block_type"] == "manual_chapter"
    assert "part of a whole" in row["content"]
    assert len(row["vector"]) == DIM
    assert math.isclose(sum(float(x) ** 2 for x in row["vector"]), 1.0)


async def test_a_dimension_mismatch_stops_the_pipeline(pipeline, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_EMBEDDING_DIM", 384)
    with pytest.raises(SystemExit, match="no client can search"):
        await pipeline.process_one("6eme", "math")
    assert not (pipeline.REGISTRY_DIR / "6eme_math_v1.parquet").exists()


async def test_a_missing_course_folder_is_skipped(pipeline):
    assert await pipeline.process_one("5eme", "history") == 0


async def test_the_manifest_carries_catalog_stamp_hashes_and_prompts(pipeline):
    await pipeline.process_one("6eme", "math")
    prompts_path = pipeline.export_prompts()
    pipeline.generate_manifest(
        {"classes": {"6eme": {"subjects": ["math"]}}},
        "https://storage.googleapis.com/test-bucket", prompts_path,
    )

    manifest = json.loads(pipeline.MANIFEST_PATH.read_text())
    assert manifest["catalog"] == {"6eme": ["math"]}
    assert manifest["embedding"] == {"model": "test/stub-embedder", "dim": DIM}
    [entry] = manifest["files"]
    assert entry["id"] == "6eme_math"
    assert entry["filename"] == "6eme_math_v1.parquet"
    assert entry["hash"].startswith("sha256:")
    assert manifest["prompts"]["url"].endswith("/prompts/prompts_v1.json")


# ── the contract with the device ─────────────────────────────────────────────

async def test_a_published_course_downloads_and_answers_a_search(
    pipeline, akili_paths, monkeypatch
):
    """Pipeline output, fed to the real download, lands searchable in L3."""
    await pipeline.process_one("6eme", "math")
    pipeline.generate_manifest(
        {"classes": {"6eme": {"subjects": ["math"]}}}, "https://example/bucket"
    )
    published = json.loads(pipeline.MANIFEST_PATH.read_text())

    install_fake_registry(
        monkeypatch, akili_paths, published,
        parquet_source=pipeline.REGISTRY_DIR / "6eme_math_v1.parquet",
    )
    ok, msg = await sync_manager.download_course("6eme", "math")
    assert ok, msg

    results = await lance_driver.search_block_index(
        list(V_A_SCALED), class_level="6eme", subject="math"
    )
    assert [r["block_id"] for r in results] == ["chapter_1_simple_fractions"]
    assert results[0]["certainty"] == pytest.approx(1.0)


# ── the shipped curriculum ───────────────────────────────────────────────────

def test_every_curriculum_entry_has_at_least_one_chapter():
    curriculum = yaml.safe_load((REGISTRY_PKG / "config" / "curriculum.yaml").read_text())
    for class_level, data in curriculum["classes"].items():
        for subject in data["subjects"]:
            chapters = [p for p in (REGISTRY_PKG / "courses" / class_level / subject).glob("*.md")
                        if p.stem != "index"]
            assert chapters, f"{class_level}/{subject} has no chapter to publish"


def test_the_prompts_the_agent_reads_are_shipped():
    """The agent reads `system_tutor` and a per-class key from prompts.json."""
    prompt_keys = {p.stem for p in (REGISTRY_PKG / "courses" / "prompts").glob("*.txt")}
    assert {"system_tutor", "6eme", "5eme"} <= prompt_keys
