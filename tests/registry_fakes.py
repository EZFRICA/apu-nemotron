"""
In-process stand-ins for the Google Cloud Storage registry, shared by the sync,
pipeline and dashboard tests. Nothing here touches the network or needs credentials.
"""

import pathlib
import shutil

import pandas as pd

from tests.conftest import DIM, V_A


def manifest(model="test/stub-embedder", dim=DIM, catalog=None, course="6eme_math"):
    """A registry manifest shaped like batch_pipeline's, with one course file."""
    class_level, subject = course.split("_", 1)
    m = {
        "catalog": catalog if catalog is not None else {class_level: [subject]},
        "files": [{"id": course, "class": class_level, "subject": subject,
                   "filename": f"{course}_v1.parquet", "hash": "sha256:x"}],
    }
    if model is not None:
        m["embedding"] = {"model": model, "dim": dim}
    return m


def course_rows(class_level="6eme", subject="math"):
    return [{
        "id": "ch1", "chapter": "ch1", "class_level": class_level, "subject": subject,
        "block_type": "manual_chapter", "content": "Les fractions.",
        "keywords": "k", "vector": list(V_A),
    }]


def install_fake_registry(monkeypatch, akili_paths, remote_manifest, rows=None,
                          parquet_source=None):
    """
    Point apu.sync.sync_manager at a fake bucket.

    The manifest fetch returns `remote_manifest`; a blob download writes `rows` as a
    parquet, or copies `parquet_source` (a file produced by the real pipeline).
    Returns the list of blob names requested, so a test can check where the client
    reads from.
    """
    from apu.sync import sync_manager

    requested = []

    async def fake_fetch_remote_json(blob_name):
        requested.append(blob_name)
        return remote_manifest

    monkeypatch.setattr(sync_manager, "_fetch_remote_json", fake_fetch_remote_json)
    monkeypatch.setattr(
        sync_manager, "LOCAL_MANIFEST_PATH",
        str(pathlib.Path(akili_paths["root"]) / "local_manifest.json"),
    )

    class _Blob:
        def __init__(self, name):
            self.name = name

        def download_to_filename(self, path):
            requested.append(self.name)
            if parquet_source is not None:
                shutil.copyfile(parquet_source, path)
            else:
                pd.DataFrame(rows if rows is not None else course_rows()).to_parquet(
                    path, index=False
                )

    class _Bucket:
        def blob(self, name):
            return _Blob(name)

    class _Client:
        def bucket(self, name):
            return _Bucket()

    async def fake_client():
        return _Client()

    monkeypatch.setattr(sync_manager, "_get_storage_client", fake_client)
    return requested


def make_registry_unreachable(monkeypatch):
    from apu.sync import sync_manager

    async def unreachable(blob_name):
        return None

    monkeypatch.setattr(sync_manager, "_fetch_remote_json", unreachable)
