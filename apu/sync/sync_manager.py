"""Device side of the cloud registry: course catalog, course download into L3, prompts.

Ported from Akili (app_local/sync/sync_manager.py). Behaviour unchanged; settings now
come from apu.config. Properties carried over as-is, worth knowing before relying on it:

  - every registry read goes through an AUTHENTICATED Google Cloud Storage client
    (service account via GOOGLE_APPLICATION_CREDENTIALS, or Application Default
    Credentials), so a device needs credentials, not only network access, to
    download a course, and the same client reads the prompts.
  - the bucket name is parsed out of REGISTRY_MANIFEST_URL (its 4th "/" segment).
  - parquets are read from the bucket root, where batch_pipeline uploads them; the
    manifest's `url` fields say /courses/, which does not match and is never read.
  - a downloaded course is checked against the sha256 the manifest states for it, since
    course content ends up in the tutor's prompt.
  - sync_with_registry refreshes the system prompts only, never the courses.
"""

import asyncio
import hashlib
import json
import os
from typing import Tuple

import pandas as pd

from apu import config
from apu.logger import get_logger
from apu.storage import lance_driver
from apu.storage.lance_driver import get_db

logger = get_logger(__name__)

# ── Local manifest path ───────────────────────────────────────────────────────
# Which courses this device holds, next to the LanceDB directory (per-device state).
LOCAL_MANIFEST_PATH = os.path.join(
    os.path.dirname(config.LANCE_DB_PATH), "local_manifest.json"
)


def _registry_bucket_name() -> str:
    # "https://storage.googleapis.com/<bucket>/manifest.json" -> "<bucket>"
    return config.REGISTRY_MANIFEST_URL.split("/")[3]


async def _get_storage_client():
    """Initializes an authenticated GCS client."""
    try:
        import google.auth
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        logger.error("Google Cloud libraries not installed.")
        return None

    gac_path = config.GOOGLE_APPLICATION_CREDENTIALS
    try:
        if gac_path and os.path.exists(gac_path):
            credentials = service_account.Credentials.from_service_account_file(gac_path)
            return storage.Client(credentials=credentials)
        else:
            credentials, project = google.auth.default()
            return storage.Client(credentials=credentials, project=project)
    except Exception as e:
        logger.error("[Sync] Auth failed: %s", e)
        return None


async def _fetch_remote_json(blob_name: str) -> dict | None:
    """Downloads and parses a remote JSON file from GCS."""
    client = await _get_storage_client()
    if not client:
        return None

    try:
        bucket = client.bucket(_registry_bucket_name())
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        logger.warning("[Sync] Error fetching %s: %s", blob_name, e)
        return None


async def get_remote_catalog() -> dict:
    """
    Fetches the remote manifest and returns the course catalog.
    """
    manifest = await _fetch_remote_json("manifest.json")
    if not manifest:
        return {}
    return manifest.get("catalog", {})


def _load_local_manifest() -> dict:
    """Loads the local manifest from disk."""
    if os.path.exists(LOCAL_MANIFEST_PATH):
        with open(LOCAL_MANIFEST_PATH) as f:
            return json.load(f)
    return {"files": {}, "last_sync": None}


def _save_local_manifest(manifest: dict):
    """Saves the local manifest to disk."""
    os.makedirs(os.path.dirname(LOCAL_MANIFEST_PATH), exist_ok=True)
    with open(LOCAL_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def is_course_available_locally(class_level: str, subject: str) -> bool:
    """
    Checks if a course already exists in local LanceDB.
    Uses the local manifest as the source of truth.
    """
    local_manifest = _load_local_manifest()
    course_id = f"{class_level}_{subject}"
    return course_id in local_manifest.get("files", {})


def cache_path_for(filename: str) -> str:
    """
    Where a downloaded course file may be written.

    The name comes from the remote manifest, which is fetched over plain HTTPS from a
    configurable URL, so it is untrusted input: "../../.bashrc" would otherwise make the
    download write outside the cache directory. Only a bare .parquet file name is accepted.
    """
    if not filename or filename != os.path.basename(filename) or os.path.isabs(filename):
        raise ValueError(f"Registry file name {filename!r} is not a plain file name.")
    if filename in (".", "..") or not filename.endswith(".parquet"):
        raise ValueError(f"Registry file name {filename!r} is not a .parquet file.")
    return os.path.join(config.CACHE_DIR, filename)


def get_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a local file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return f"sha256:{sha256_hash.hexdigest()}"


async def download_course(class_level: str, subject: str) -> Tuple[bool, str]:
    """
    Downloads a specific course from the registry and imports it into LanceDB.
    """
    logger.info(f"[Sync] Downloading course: {class_level}/{subject}...")

    # 1. Fetch remote manifest
    manifest = await _fetch_remote_json("manifest.json")
    if not manifest:
        return False, "Unable to reach the registry. Check your credentials."

    # 2. Find the course file in the manifest
    course_id = f"{class_level}_{subject}"
    file_info = None
    for f in manifest.get("files", []):
        if f.get("id") == course_id or (
            f.get("class") == class_level and f.get("subject") == subject
        ):
            file_info = f
            break

    if not file_info:
        return False, f"Course '{class_level}/{subject}' not found in the registry."

    # 2b. Refuse a registry built with a different embedder BEFORE downloading.
    # The manifest carries the stamp, so this is knowable now rather than at the
    # student's first question, and it saves pulling a parquet that cannot be
    # searched anyway.
    remote = manifest.get("embedding")
    if remote:
        if (remote.get("dim") != config.LOCAL_EMBEDDING_DIM
                or remote.get("model") != config.LOCAL_EMBEDDING_MODEL):
            return False, (
                f"This registry was built with '{remote.get('model')}' at "
                f"{remote.get('dim')} dimensions, but this device is configured "
                f"for '{config.LOCAL_EMBEDDING_MODEL}' at {config.LOCAL_EMBEDDING_DIM}. "
                f"Searching it would return noise. Regenerate the registry with "
                f"the configured model, or change LOCAL_EMBEDDING_MODEL to match."
            )
    else:
        logger.warning("[Sync] Registry manifest carries no embedding stamp — "
              "it predates stamping. Import will be checked by dimension only.")

    # 3. Download the parquet file via GCS Client
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    try:
        temp_parquet = cache_path_for(file_info.get("filename", ""))
    except ValueError as error:
        return False, f"Refused by the device: {error}"

    try:
        client = await _get_storage_client()
        bucket = client.bucket(_registry_bucket_name())
        # The parquet files are at the root of the bucket, not in /courses
        blob_name = file_info["filename"]
        blob = bucket.blob(blob_name)
        blob.download_to_filename(temp_parquet)
    except Exception as e:
        return False, f"GCS Download failed: {e}"

    # The manifest states a sha256 per file; check it instead of merely recording it. This
    # is what makes the manifest an integrity statement rather than a label: course content
    # is read into the tutor's prompt, so a file altered in the bucket or in flight would
    # otherwise be imported and used without a word.
    expected_hash = file_info.get("hash")
    if expected_hash:
        actual_hash = get_file_hash(temp_parquet)
        if actual_hash != expected_hash:
            os.remove(temp_parquet)
            logger.error("[Sync] Hash mismatch for %s: manifest %s, downloaded %s",
                         blob_name, expected_hash, actual_hash)
            return False, (
                f"Refused: '{blob_name}' does not match the hash in the registry manifest. "
                "The file was not imported."
            )
    else:
        logger.warning("[Sync] Manifest carries no hash for %s; imported unverified.", blob_name)

    # 4-5. Import into LanceDB and record it in the local manifest
    try:
        import_course_parquet(temp_parquet, class_level, subject, file_info.get("hash", ""))
        # Cleanup temp file
        os.remove(temp_parquet)
    except Exception as e:
        return False, f"Import into LanceDB failed: {e}"

    return True, f"Course '{class_level} — {subject}' successfully downloaded and imported."


def import_course_parquet(parquet_path: str, class_level: str, subject: str, file_hash: str = "") -> int:
    """
    Import one course parquet into the local L3 registry and record it as downloaded.

    Extracted from download_course, unchanged, so the demo preparation can import a
    registry built locally (cloud_registry/registry/) through the exact same path as a
    course downloaded from the bucket. Returns the number of rows imported.
    """
    df = pd.read_parquet(parquet_path)
    db = get_db()

    # list_table_names, never db.list_tables(): on lancedb 0.30.2 the latter
    # returns a response model whose __contains__ never matches, so the
    # replace branch below was dead in Akili.
    all_tables = lance_driver.list_table_names(db)
    if "edu_registry" not in all_tables:
        try:
            db.create_table("edu_registry", data=df)
        except Exception:
            # If it was created by another thread just in time, just open it
            table = db.open_table("edu_registry")
            table.delete(f"class_level = '{class_level}' AND subject = '{subject}'")
            table.add(df)
    else:
        table = db.open_table("edu_registry")
        # Replace old entries for this class/subject
        table.delete(f"class_level = '{class_level}' AND subject = '{subject}'")
        table.add(df)

    # Record which embedder produced these vectors, so a later model change
    # is refused with a readable message instead of degrading silently.
    lance_driver.write_stamp("edu_registry")

    local_manifest = _load_local_manifest()
    local_manifest["files"][f"{class_level}_{subject}"] = {
        "class": class_level,
        "subject": subject,
        "hash": file_hash,
        "downloaded_at": __import__("datetime").datetime.now().isoformat()
    }
    _save_local_manifest(local_manifest)
    return len(df)


async def sync_with_registry() -> Tuple[bool, str]:
    """
    Refresh the system prompts from the registry.

    In Akili this was an inline re-implementation of download_prompts that read a
    flag never assigned on three paths, including the steady state where
    everything is already current, so "Check for Updates" raised on every press
    after the first. The duplicate is gone rather than patched.
    """
    logger.info("Starting synchronization with Akili registry...")

    prompts_path = os.path.join(
        os.path.dirname(config.LANCE_DB_PATH), "prompts.json"
    )
    manifest = await _fetch_remote_json("manifest.json")
    if not manifest:
        return False, "Unable to reach the registry (check your connection)."

    if "prompts" not in manifest:
        return True, "Registry has no prompts section — nothing to update."

    remote_hash = manifest["prompts"].get("hash")
    if (remote_hash and os.path.exists(prompts_path)
            and get_file_hash(prompts_path) == remote_hash):
        return True, "Everything is up to date."

    prompts_data = await _fetch_remote_json("prompts/prompts_v1.json")
    if not prompts_data:
        return False, "Could not download the system prompts."

    os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)

    return True, "System prompts updated."


if __name__ == "__main__":
    asyncio.run(sync_with_registry())
