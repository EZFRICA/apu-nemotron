"""Central configuration: model IDs, endpoints, and tunables.

Kept in one place so swapping a model size (e.g. Nano instead of Super for the main
call, to trade reasoning quality for latency and credit budget) is a one-line change
rather than a search-and-replace across the codebase.

Modules read these as `config.NAME` at call time rather than binding them with
`from apu.config import NAME`, so a test can redirect storage paths or the embedder
with monkeypatch and every call site sees the change.
"""

import os

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

# Larger model for the answer the user reads.
MAIN_MODEL = os.environ.get("NEBIUS_MAIN_MODEL", "nvidia/nemotron-3-super-120b-a12b")

# Smaller, cheaper model for the background extraction call.
EXTRACTION_MODEL = os.environ.get("NEBIUS_EXTRACTION_MODEL", "nvidia/nemotron-3-nano-30b-a3b")

# Local embedder, swappable depending on target device. paraphrase-multilingual-MiniLM-L12-v2
# (384 dim) is the current default, chosen for low-capacity hardware; a beefier machine
# could use a larger multilingual model without touching any other part of the pipeline.
#
# The value must be the full fastembed id. fastembed rejects the bare
# "paraphrase-multilingual-MiniLM-L12-v2" as unsupported (checked against fastembed
# 0.8.0), so the scaffold's original default could never load.
LOCAL_EMBEDDING_MODEL = os.environ.get(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Must match the model above. The L3 driver refuses to search a table whose vectors
# have another width: a vector-space mismatch does not fail on its own, it returns
# noise ranked as though it were relevant.
LOCAL_EMBEDDING_DIM = int(os.environ.get("LOCAL_EMBEDDING_DIM", "384"))

# Where the ONNX files live on disk. The embedder never downloads at runtime, because
# on a disconnected device a download attempt hangs instead of failing. Populate it
# once on a connected machine with scripts/fetch_embedding_model.py and ship it with
# the deployment.
LOCAL_EMBEDDING_CACHE_DIR = os.environ.get(
    "LOCAL_EMBEDDING_CACHE_DIR", os.path.join(_REPO_ROOT, "models")
)

# ── Local storage (L2 / L3) ──────────────────────────────────────────────────
# Per-device state, not source. Kept under a gitignored directory so a fresh clone
# does not inherit another machine's memory.
DATA_DIR = os.environ.get("APU_DATA_DIR", os.path.join(_REPO_ROOT, "data"))

# L3: LanceDB vector store (course registry `edu_registry` + archived `user_memory`).
LANCE_DB_PATH = os.path.join(DATA_DIR, "akili_db")

# L2: the DLL chain (prev/next pointers plus node content), persisted as JSON.
METADATA_LINKS_PATH = os.path.join(DATA_DIR, "memory", "metadata_links.json")

# Sidecar recording which embedder wrote each L3 table, so a model swap is caught.
EMBEDDING_STAMP_PATH = os.path.join(DATA_DIR, "embedding_stamp.json")

# ── MMU tunables ─────────────────────────────────────────────────────────────
# NOT WIRED. The scaffold documents a hard cap of 12 active blocks. That figure comes
# from the travel-agent APU (4 fixed + 8 dynamic; its controller says "the absolute
# maximum blocks we can ever send is 12"). The Akili code this repo was ported from
# caps at 4 fixed + MAX_DYNAMIC_BLOCKS below. The port keeps Akili's behaviour until
# the conflict is decided; see HACKATHON.md.
MAX_ACTIVE_BLOCKS = int(os.environ.get("APU_MAX_ACTIVE_BLOCKS", "12"))

# Cap on non-fixed blocks in the DLL. Creating one more at the cap pages the least
# recently accessed non-fixed block out to L3. 5 is Akili's value.
MAX_DYNAMIC_BLOCKS = int(os.environ.get("APU_MAX_DYNAMIC_BLOCKS", "5"))

# ── Retrieval thresholds ─────────────────────────────────────────────────────
# Minimum cosine similarity for a block to enter the working context.
#
# THESE ARE CALIBRATED TO THE EMBEDDING MODEL and must be re-measured if it changes.
# Measured in Akili with paraphrase-multilingual-MiniLM-L12-v2 on real course content,
# query "Explique-moi les fractions":
#
#   chapter_1_simple_fractions      0.670   <- the right chapter
#   chapter_2_decimal_numbers       0.507   <- same subject, related
#   chapter_3_angles_and_geometry   0.284   <- unrelated
#
# The earlier values (0.70/0.75/0.80) rejected ALL of them, including the exact match.
# Ordering follows the original design: fondamental (always relevant) < cours < temp
# (most recent context, most selective).
CERTAINTY_THRESHOLDS = {
    "fondamental":    0.40,   # student_profile / learning_preferences
    "cours":          0.45,   # active_course
    "manual_chapter": 0.45,   # what every shipped course row actually carries
    "temp":           0.50,   # current_session
}
MIN_RELEVANCE_CERTAINTY = 0.45

# ── Education defaults (Akili) ───────────────────────────────────────────────
EDU_DEFAULT_CLASS = "6eme"
EDU_DEFAULT_SUBJECT = "math"
