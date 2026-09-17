"""
Batch Pipeline — Akili Registry
=================================
Reads curriculum.yaml and processes all class/subject combinations in one run:
  1. MD files → local ONNX embeddings → .parquet files (in cloud_registry/registry/)
  2. prompts → prompts/prompts_v1.json
  3. manifest.json (with catalog, embedding stamp and per-file hashes)
  4. (Optional) Upload all files to Google Cloud Storage

Ported from Akili (cloud_registry/pipeline/batch_pipeline.py). Behaviour unchanged,
except that the embedder is built on first use instead of at import: built with
allow_download=True at import, merely importing this module (a test, an IDE) would
download ~240MB.

Usage:
    uv run python cloud_registry/pipeline/batch_pipeline.py

Upload to GCS (requires GCS_BUCKET_NAME, and GOOGLE_APPLICATION_CREDENTIALS or
Application Default Credentials):
    uv run python cloud_registry/pipeline/batch_pipeline.py --upload
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Path alignment: run as a script, the repo root is not on sys.path.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apu import config  # noqa: E402
from apu.embeddings.local_embedder import (  # noqa: E402
    build_embedder,
    embedding_stamp,
    normalize_vector,
)
from cloud_registry.config import settings as cloud_settings  # noqa: E402

# ── Config paths ──────────────────────────────────────────────────────────────
PIPELINE_DIR   = Path(__file__).resolve().parent
REGISTRY_DIR   = PIPELINE_DIR.parent / "registry"
COURSES_DIR    = PIPELINE_DIR.parent / "courses"
CURRICULUM_PATH = PIPELINE_DIR.parent / "config" / "curriculum.yaml"
MANIFEST_PATH  = REGISTRY_DIR / "manifest.json"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedder (shared across all tasks) ───────────────────────────────────────
# The SAME model the client queries with: both read apu.config. In Akili this was
# once GoogleGenerativeAIEmbeddings against a hardcoded model id, which is how the
# registry ended up published at 3072 dimensions while the client queried at 384.
# allow_download=True because the pipeline runs on a connected machine by
# definition; the client never gets that flag.
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = build_embedder(allow_download=True)
    return _embedder


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate embeddings and parquet for one class/subject
# ─────────────────────────────────────────────────────────────────────────────

async def process_one(class_level: str, subject: str) -> int:
    """
    Reads all .md files for class_level/subject,
    embeds them with the local embedder, and saves a .parquet file.
    Returns the number of blocks processed.
    """
    subject_path = COURSES_DIR / class_level / subject
    if not subject_path.exists():
        print(f"  [SKIP] No folder found: {subject_path}")
        return 0

    blocks = []
    md_files = sorted(subject_path.glob("*.md"))

    if not md_files:
        print(f"  [SKIP] No .md files in {subject_path}")
        return 0

    print(f"  Processing {len(md_files)} files for {class_level}/{subject}...")

    for md_file in md_files:
        if md_file.stem == "index":
            continue  # Skip index files

        chapter_id = md_file.stem
        content    = md_file.read_text(encoding="utf-8")

        keywords_text = (
            f"Level: {class_level} | Subject: {subject} | "
            f"Chapter: {chapter_id.replace('_', ' ')}"
        )
        embed_input = f"{keywords_text}\n\n{content[:2000]}"

        try:
            vector = await _get_embedder().aembed_query(embed_input)
        except Exception as e:
            print(f"    [ERROR] Embedding failed for {chapter_id}: {e}")
            continue

        # Normalise before publishing: the client's certainty scale
        # (1 - distance/2) is cosine similarity only if BOTH sides are unit
        # length, and downloaded parquets go into LanceDB verbatim.
        vector = normalize_vector(vector)

        if len(vector) != config.LOCAL_EMBEDDING_DIM:
            raise SystemExit(
                f"Embedder produced {len(vector)} dimensions but "
                f"LOCAL_EMBEDDING_DIM is {config.LOCAL_EMBEDDING_DIM}. Publishing "
                f"this would ship a registry no client can search. Fix "
                f"LOCAL_EMBEDDING_MODEL / LOCAL_EMBEDDING_DIM before continuing."
            )

        blocks.append({
            "id":          chapter_id,
            "chapter":     chapter_id,
            "class_level": class_level,
            "subject":     subject,
            "block_type":  "manual_chapter",
            "content":     content,
            "keywords":    keywords_text,
            "vector":      vector,
        })
        print(f"    [OK] {chapter_id} (dim={len(vector)})")

    if not blocks:
        return 0

    df = pd.DataFrame(blocks)
    output_filename = f"{class_level}_{subject}_v1.parquet"
    output_path = REGISTRY_DIR / output_filename
    df.to_parquet(output_path, index=False)
    print(f"  ✅ Saved: {output_path} ({len(blocks)} blocks)")
    return len(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Export prompts
# ─────────────────────────────────────────────────────────────────────────────

def export_prompts() -> str:
    """Packs all .txt files in courses/prompts/ into prompts_v1.json."""
    prompts_dir = COURSES_DIR / "prompts"
    if not prompts_dir.exists():
        print("  [SKIP] No prompts directory found.")
        return ""

    import json
    prompts_data = {}
    for txt_file in prompts_dir.glob("*.txt"):
        prompts_data[txt_file.stem] = txt_file.read_text(encoding="utf-8")

    if not prompts_data:
        print("  [SKIP] No .txt prompts found.")
        return ""

    prompts_output = REGISTRY_DIR / "prompts" / "prompts_v1.json"
    prompts_output.parent.mkdir(parents=True, exist_ok=True)
    prompts_output.write_text(
        json.dumps(prompts_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✅ Prompts saved: {prompts_output} ({len(prompts_data)} entries)")
    return str(prompts_output)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Generate manifest.json (client-compatible format)
# ─────────────────────────────────────────────────────────────────────────────

def generate_manifest(curriculum: dict, base_url: str, prompts_path: str = "") -> None:
    """
    Generates manifest.json with:
      - catalog: {class: [subjects]}        ← used by the dashboard for selectboxes
      - embedding: {model, dim}             ← checked by the device before a download
      - files: [{id, class, subject, url, hash, size_bytes, ...}]
      - prompts: {url, hash}

    NOTE (carried over from Akili): `url` says /courses/<file>, but upload_to_gcs
    uploads parquets at the bucket root, and the client downloads them from the
    root. The client never reads `url`, so nothing breaks, but the field is wrong.
    """
    import hashlib
    import json

    def sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    # Build catalog from curriculum
    catalog = {
        cls: list(data["subjects"])
        for cls, data in curriculum["classes"].items()
    }

    files_list = []
    for parquet_file in sorted(REGISTRY_DIR.glob("*.parquet")):
        parts = parquet_file.stem.split("_")  # e.g. ["6eme", "math", "v1"]
        if len(parts) < 2:
            continue

        # Reconstruct class and subject (handles multi-word subjects like "histoire_geo")
        class_level = parts[0]
        version     = parts[-1]         # last part is version
        subject     = "_".join(parts[1:-1])  # everything between

        file_entry = {
            "id":          f"{class_level}_{subject}",
            "class":       class_level,
            "subject":     subject,
            "version":     version,
            "filename":    parquet_file.name,
            "url":         f"{base_url}/courses/{parquet_file.name}",
            "hash":        sha256(parquet_file),
            "size_bytes":  parquet_file.stat().st_size,
            "updated_at":  datetime.now().isoformat(),
        }
        files_list.append(file_entry)

    manifest = {
        "version":      datetime.now().strftime("%Y%m%d%H%M"),
        "generated_at": datetime.now().isoformat(),
        # Which vector space these parquets live in. The client compares this
        # against its own configuration and refuses to download on a mismatch,
        # instead of returning noise ranked as though it were relevant.
        "embedding":    embedding_stamp(),
        "catalog":      catalog,
        "files":        files_list,
    }

    # Add prompts info if available
    if prompts_path and Path(prompts_path).exists():
        manifest["prompts"] = {
            "url":  f"{base_url}/prompts/prompts_v1.json",
            "hash": sha256(prompts_path),
        }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✅ Manifest: {MANIFEST_PATH} ({len(files_list)} courses, catalog: {list(catalog.keys())})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Upload to GCS (optional)
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_gcs(bucket_name: str) -> None:
    """
    Uploads all registry files to Google Cloud Storage.
    Supports both GOOGLE_APPLICATION_CREDENTIALS and Application Default Credentials.
    """
    try:
        import google.auth
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        print("  [ERROR] Google Cloud libraries not installed.")
        print("          Run: uv sync")
        return

    # ── Authentication Logic ──────────────────────────────────────────
    gac_path = cloud_settings.GOOGLE_APPLICATION_CREDENTIALS

    try:
        if gac_path and os.path.exists(gac_path):
            print(f"  🔑 Using Service Account: {gac_path}")
            credentials = service_account.Credentials.from_service_account_file(gac_path)
            client = storage.Client(credentials=credentials)
        else:
            print("  🔑 Using Application Default Credentials (ADC)")
            credentials, project = google.auth.default()
            client = storage.Client(credentials=credentials, project=project)

        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"  [ERROR] Authentication failed: {e}")
        return

    files_to_upload = list(REGISTRY_DIR.rglob("*"))
    for local_path in files_to_upload:
        if local_path.is_dir():
            continue
        relative = local_path.relative_to(REGISTRY_DIR)
        blob_name = str(relative).replace("\\", "/")
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        print(f"  ☁️  Uploaded: gs://{bucket_name}/{blob_name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main(upload: bool = False):
    # Load curriculum
    with open(CURRICULUM_PATH, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)

    classes = curriculum.get("classes", {})
    print(f"\n{'='*60}")
    print(f"  AKILI BATCH PIPELINE")
    print(f"  Classes: {list(classes.keys())}")
    print(f"{'='*60}\n")

    # ── Step 1: Generate parquets ─────────────────────────────────────
    print("STEP 1 — Generating course parquets...\n")
    total_blocks = 0
    for class_level, class_data in classes.items():
        for subject in class_data.get("subjects", []):
            print(f"▶ {class_level}/{subject}")
            n = await process_one(class_level, subject)
            total_blocks += n

    print(f"\n  Total blocks generated: {total_blocks}\n")

    # ── Step 2: Export prompts ────────────────────────────────────────
    print("STEP 2 — Exporting prompts...\n")
    prompts_path = export_prompts()

    # ── Step 3: Generate manifest ─────────────────────────────────────
    print("\nSTEP 3 — Generating manifest.json...\n")
    bucket_name = cloud_settings.GCS_BUCKET_NAME
    base_url    = f"https://storage.googleapis.com/{bucket_name}"
    generate_manifest(curriculum, base_url, prompts_path)

    # ── Step 4: Upload to GCS (optional) ─────────────────────────────
    if upload:
        print("\nSTEP 4 — Uploading to Google Cloud Storage...\n")
        if not bucket_name:
            print("  [ERROR] GCS_BUCKET_NAME not set in .env")
        else:
            upload_to_gcs(bucket_name)
    else:
        print("\nSTEP 4 — GCS upload skipped. Run with --upload to push to cloud.\n")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Registry files are in: {REGISTRY_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Akili Registry Batch Pipeline")
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload generated files to Google Cloud Storage"
    )
    args = parser.parse_args()
    asyncio.run(main(upload=args.upload))
