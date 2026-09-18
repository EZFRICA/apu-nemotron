"""
manifest_generator.py
=====================
Standalone manifest generator. Ported unchanged from Akili.
Scans the registry/ folder and produces manifest.json in
the client-compatible format (with 'catalog' and per-file 'class'/'subject' fields).

NOTE (carried over from Akili, not fixed in the port): unlike batch_pipeline's
generate_manifest, this one writes NO `embedding` stamp. A manifest regenerated here
makes devices skip the model check and import on dimension alone (they print a
warning). Prefer running batch_pipeline.py.

Usage:
    uv run python cloud_registry/pipeline/manifest_generator.py
    uv run python cloud_registry/pipeline/manifest_generator.py my-bucket-name
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PIPELINE_DIR    = Path(__file__).resolve().parent
REGISTRY_DIR    = PIPELINE_DIR.parent / "registry"
CURRICULUM_PATH = PIPELINE_DIR.parent / "config" / "curriculum.yaml"
MANIFEST_PATH   = REGISTRY_DIR / "manifest.json"


def sha256(filepath: str) -> str:
    """Calculates the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def generate_manifest(base_url: str = "https://storage.googleapis.com/akili-registry"):
    """
    Scans registry/ for .parquet files and builds manifest.json.
    Includes 'catalog' field for the interface's course selection.
    """
    # Load curriculum to build the catalog
    catalog = {}
    if CURRICULUM_PATH.exists():
        with open(CURRICULUM_PATH, encoding="utf-8") as f:
            curriculum = yaml.safe_load(f)
        catalog = {
            cls: list(data.get("subjects", []))
            for cls, data in curriculum.get("classes", {}).items()
        }
    else:
        print("  [WARN] curriculum.yaml not found — catalog will be empty.")

    files_list = []

    for parquet_file in sorted(REGISTRY_DIR.glob("*.parquet")):
        parts = parquet_file.stem.split("_")  # e.g. ["6eme", "math", "v1"]
        if len(parts) < 3:
            continue

        class_level = parts[0]
        version     = parts[-1]                 # last part is version
        subject     = "_".join(parts[1:-1])     # everything in between

        file_entry = {
            "id":          f"{class_level}_{subject}",
            "class":       class_level,
            "subject":     subject,
            "version":     version,
            "filename":    parquet_file.name,
            "url":         f"{base_url}/courses/{parquet_file.name}",
            "hash":        sha256(str(parquet_file)),
            "size_bytes":  parquet_file.stat().st_size,
            "updated_at":  datetime.now().isoformat(),
        }
        files_list.append(file_entry)

    manifest = {
        "version":      datetime.now().strftime("%Y%m%d%H%M"),
        "generated_at": datetime.now().isoformat(),
        "catalog":      catalog,   # ← Used by the interface for class/subject selection
        "files":        files_list,
    }

    # Add prompts section if file exists
    prompts_file = REGISTRY_DIR / "prompts" / "prompts_v1.json"
    if prompts_file.exists():
        manifest["prompts"] = {
            "url":  f"{base_url}/prompts/prompts_v1.json",
            "hash": sha256(str(prompts_file)),
        }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Success! Manifest generated at {MANIFEST_PATH}")
    print(f"  → {len(files_list)} course files")
    print(f"  → Catalog: {catalog}")


if __name__ == "__main__":
    bucket = sys.argv[1] if len(sys.argv) > 1 else "akili-registry"
    base_url = f"https://storage.googleapis.com/{bucket}"
    generate_manifest(base_url)
