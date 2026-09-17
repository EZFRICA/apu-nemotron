# APU on Nemotron

A tiered, persistent memory architecture for LLM agents, running on NVIDIA Nemotron served through Nebius Token Factory.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/), Personal AI Track.

## What this is

This repository is the Nebius Token Factory / NVIDIA Nemotron port of the Agent Processor Unit (APU), a four-tier memory hierarchy for AI agents (L1 in-process cache, L2 Redis working-context list, L3 Weaviate vector store, L4 Letta long-term archive), with a BMJ (Bidirectional Metadata Jump) routing algorithm and an async priority-queue scheduler.

See [HACKATHON.md](./HACKATHON.md) for what is new in this repository versus the pre-existing APU / Akili work, as required by the hackathon rules.

## Why

The architecture targets agents that need to run useful workloads on constrained hardware and inconsistent connectivity, the kind of machine class and budget you find in a typical West African classroom or field deployment, not a high-end workstation. Two design choices follow from that constraint:

- The embedding model stays local (`paraphrase-multilingual-MiniLM-L12-v2` via fastembed/ONNX) rather than calling a remote embedding API, since it is the component on the hot path for every retrieval. It is swappable: pick a smaller or larger embedder depending on the target device.
- The main reasoning call and the background memory-extraction call are split across two Nemotron sizes, so the always-on agent stays responsive without burning through inference budget on every turn.

## How

```
User turn
   │
   ▼
Planner node (LangGraph) ──► MMU routes query across L1 → L2 → L3 → L4
   │                                     │
   ▼                                     ▼
Main answer call                 Local embedder (fastembed/ONNX)
(Nemotron, Token Factory)        for retrieval and block matching
   │
   ▼
Answer returned to user
   │
   ▼ (background, off the critical path)
Extraction call (Nemotron Nano, Token Factory)
   │
   ▼
New / updated memory blocks written to L2, paged to L3 on eviction
```

## Model routing

| Call | Model | Rationale |
|---|---|---|
| Main answer | `nvidia/nemotron-3-super-120b-a12b` | Higher-accuracy reasoning for the response the user actually sees |
| Background extraction | `nvidia/nemotron-3-nano-30b-a3b` | Cheap, fast, off the critical path, historically the largest source of latency and variance in this pipeline |

Both are served through the Nebius Token Factory OpenAI-compatible API. See `apu/inference/nebius_client.py`.

## Quickstart

Every command runs through [uv](https://docs.astral.sh/uv/), from the repository root. There is no virtual environment to create or activate by hand: `uv run` uses the project's `.venv/` automatically.

### 1. Prerequisites

- **uv**. Install it with one of:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  ```bash
  brew install uv
  ```
  uv downloads Python 3.13 by itself if it is not already on the machine.
- **A Nebius Token Factory API key**, from https://tokenfactory.nebius.com/ (hackathon credits: see the comment at the top of `.env.example`).
- **About 300 MB of disk** for the local embedding model, and a network connection for this first setup. After setup, only the Nemotron calls need the network.
- **liblouis**, only for braille output: `brew install liblouis` (macOS) or `apt install liblouis20` (Debian/Ubuntu). Everything else works without it.

### 2. Get the code and install the dependencies

```bash
git clone git@github.com:EZFRICA/apu-nemotron.git
```

```bash
cd apu-nemotron
```

```bash
uv sync
```

`uv sync` creates `.venv/` and installs the exact versions pinned in `uv.lock`, including the test tools and Streamlit.

### 3. Configure

```bash
cp .env.example .env
```

Then open `.env` and set `NEBIUS_API_KEY`. That is the only required value; everything else has a working default.

| Variable | Default | What it does |
|---|---|---|
| `NEBIUS_API_KEY` | *(none, required)* | Token Factory key used by both Nemotron calls |
| `NEBIUS_MAIN_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Model that writes the answer the student reads |
| `NEBIUS_EXTRACTION_MODEL` | `nvidia/nemotron-3-nano-30b-a3b` | Model that extracts what to remember from each exchange |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Local ONNX embedder (full fastembed id) |
| `LOCAL_EMBEDDING_DIM` | `384` | Vector width; must match the embedder |
| `LOCAL_EMBEDDING_CACHE_DIR` | `./models` | Where the ONNX model files are read from |
| `APU_DATA_DIR` | `./data` | Per-device state: DLL (L2), LanceDB (L3), embedding stamp |
| `APU_MAX_DYNAMIC_BLOCKS` | `5` | Cap on non-fixed memory blocks before LRU page-out to L3 |
| `REGISTRY_MANIFEST_URL` | `https://storage.googleapis.com/akili-registry/manifest.json` | Cloud registry the device downloads courses from (bucket read from the URL) |
| `GOOGLE_APPLICATION_CREDENTIALS` | *(unset: Application Default Credentials)* | Service-account JSON with access to the registry bucket |
| `GCS_BUCKET_NAME` | *(none)* | Only for publishing: bucket `batch_pipeline.py --upload` writes to |
| `TAVILY_API_KEY` | *(none)* | Web search; credits come with the hackathon promo bundle |
| `APU_CLASS_POLICIES_PATH` | `registries/class_policies.json` | Per-class policy: escalation threshold, extra excluded search domains |
| `APU_TEACHER_ASSIGNMENTS_PATH` | `registries/teacher_assignments.json` | Who teaches or administers which class; the only source of roles |
| `APU_STUDENT_ID` / `APU_CLASS_ID` | `eleve-demo` / `lycee-cocody:3eA` | Identity the dashboard uses for every session (stub, no login) |

### 4. Fetch the embedding model (once, needs the network)

```bash
uv run python scripts/fetch_embedding_model.py
```

This downloads about 240 MB into `./models` and then checks that the model loads from that cache with no network. The app itself **never downloads at runtime**: on an offline device, copy `models/` over from a connected machine, or point `LOCAL_EMBEDDING_CACHE_DIR` at an existing copy.

### 5. Course content: the cloud registry

Course chapters reach the tutor through a **cloud registry**: Markdown courses are embedded once with the same local model, published as Parquet files plus a `manifest.json` to a Google Cloud Storage bucket, and each device downloads the courses it needs into its local LanceDB (L3). No LLM is involved on this side.

**On a device (download courses).** The client reads the bucket through an authenticated Google Cloud client, so the device needs credentials, not only network access. Either sign in with your Google account (the `gcloud` CLI is a Google tool, not a Python command):

```bash
gcloud auth application-default login
```

or set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to a service-account JSON with read access to the bucket. Then, in the interface (step 7), pick a grade and a subject in the sidebar and click **Download & Activate**. **Check for Updates** refreshes the system prompts from the registry. Courses stay usable offline once downloaded.

**Publishing your own registry (maintainer).** Chapters live in `cloud_registry/courses/<grade>/<subject>/*.md`, the list of grades and subjects in `cloud_registry/config/curriculum.yaml`, the tutor prompts in `cloud_registry/courses/prompts/`. Build the registry locally into `cloud_registry/registry/` to inspect it:

```bash
uv run python cloud_registry/pipeline/batch_pipeline.py
```

Then set `GCS_BUCKET_NAME` in `.env`, authenticate as above with write access, and publish:

```bash
uv run python cloud_registry/pipeline/batch_pipeline.py --upload
```

Point devices at it with `REGISTRY_MANIFEST_URL=https://storage.googleapis.com/<your-bucket>/manifest.json`. The registry and the devices must use the same embedding model: a device refuses to download from a registry built with another one. Details and known issues: [cloud_registry/README.md](./cloud_registry/README.md).

### 6. Run the tests

```bash
uv run pytest
```

The suite needs neither the API key nor the network: the Nebius client is replaced by a fake that also checks which model each call goes to. If step 4 was skipped, the 8 tests that use the real embedding model are reported as skipped and everything else still runs.

### 7. Launch the interface

```bash
uv run streamlit run apu/ui/dashboard.py
```

Streamlit prints a local URL, `http://localhost:8501` by default; open it in a browser. On the very first run Streamlit may ask for an email address in the terminal: press Enter to skip.

What you get:

- **Right, the chat.** Each question goes to Nemotron Super for the answer, then to Nemotron Nano for memory extraction. Both calls finish before the answer is displayed, so a turn takes the time of the two calls together.
- **Left, the memory hierarchy while it works.** L1 shows the blocks hot in RAM with their hit rate, L2 the DLL chain in HEAD → TAIL order with each block's type, L3 the LanceDB tables and their row counts.
- **Sidebar.** Course selection from the cloud registry (**Download & Activate**, or **Activate** for a course already on the device), the number of past exchanges sent to the model (0 means memory only), **Check for Updates** and **Reset Memory**. When the registry cannot be reached, the courses already downloaded are still offered.

Useful variants:

```bash
uv run streamlit run apu/ui/dashboard.py --server.port 8502
```

```bash
uv run streamlit run apu/ui/dashboard.py --server.headless true
```

The first picks another port; the second is for a remote machine or server, where Streamlit should not try to open a browser.

### 8. Where state lives, and how to reset it

Everything the device holds is under `data/` (gitignored): `data/memory/metadata_links.json` is the DLL (L2), `data/akili_db/` the LanceDB store (L3: downloaded courses in `edu_registry`, archived student memory in `user_memory`), `data/local_manifest.json` the list of downloaded courses, `data/prompts.json` the system prompts from the registry. **Reset Memory** in the sidebar wipes L1 and L2; rows already archived in L3 stay. To start completely from scratch, stop Streamlit and delete the directory:

```bash
rm -rf data
```

### Troubleshooting

- **`Inference failed: NEBIUS_API_KEY is not set` in the chat.** Set the key in `.env`, then stop Streamlit (Ctrl+C) and launch it again.
- **`Embedding model ... is not present in ...`.** Step 4 was not run, or `LOCAL_EMBEDDING_CACHE_DIR` points to the wrong place.
- **"Registry unreachable" in the sidebar.** The device could not read the registry bucket: no Google credentials (step 5), no network, or a wrong `REGISTRY_MANIFEST_URL`. The terminal running Streamlit prints the exact reason (`[Sync] Auth failed: ...`). Courses already downloaded remain available.
- **"This registry was built with '...'" when downloading.** The registry and this device use different embedding models; align `LOCAL_EMBEDDING_MODEL` with the registry's, or republish the registry.
- **`Port 8501 is already in use`.** Use `--server.port` as shown above.
- **`Inference failed: The topical guard could not classify this turn`.** Every question first goes through the topical guard, which calls Nemotron on Token Factory. Same fix as a missing or invalid `NEBIUS_API_KEY`. The tutor deliberately does not answer when the guard cannot run.

## Guardrails, escalations and the teacher API

### Topical guard (NeMo Guardrails)

Every student turn goes through one shared input rail before anything else (`apu/guardrails/config/`, Colang). A classifier running on Nemotron Super through Token Factory decides **strictly school use** (lessons, exercises, revision, study-related research) versus **off-topic**. There is no list of allowed subjects.

What varies per class is data, not Colang: `registries/class_policies.json` holds each class's `escalation_threshold` and extra search exclusions. A session reads its class policy **once, when it opens**; a changed threshold applies to the next session.

- Off-topic, below the threshold: a kind redirect towards schoolwork.
- Off-topic, reaching the threshold: a firmer reply, and one `EscalationEvent` written in the background. Further attempts in the same session stay firm without adding events.
- The off-topic counter lives in memory for the session and restarts on reconnection.
- If the classifier cannot be reached, the turn is not answered.

### Web search (Tavily)

`apu/tools/web_search.py` only searches for a turn the topical guard validated, excludes social networks for everyone (`GLOBAL_EXCLUDED_DOMAINS`) plus each class's own additions, and returns its sources. `apu/modality/citations.py` renders them per output channel: a list at the end in text or braille; source names said aloud (never URLs) in voice, plus the written list when a screen is available. **Search is not yet called during answers**: how the model requests it waits on the tool-calling smoke test (see [HACKATHON.md](./HACKATHON.md)).

### Escalations and clustering

Escalation events are stored apart from the tutoring memory (`data/escalations.sqlite3`): they are never DLL blocks, never L3 rows, and never returned by the pedagogical search. Events are immutable; "resolved" is a separate record joined at read time. After every 5 new events in a class (`APU_ESCALATION_CLUSTER_TRIGGER_COUNT`), a background job clusters that class's events (HDBSCAN on local embeddings); reads only return the last stored result.

### Teacher and admin API (FastAPI)

```bash
uv run uvicorn apu.api.app:app --reload
```

| Route | Who |
|---|---|
| `GET /escalations?class_id=...` | the class's teacher, or an admin of its establishment |
| `POST /escalations/{event_id}/resolve` | same; body `{"note": "..."}` is optional |
| `GET /escalations/clusters?class_id=...` | same |
| `GET /establishments/{establishment_id}/classes` | admin: every class of the establishment; teacher: their own |

Roles and scopes come only from `registries/teacher_assignments.json`, never from the request. With the demo registries (the `curl` CLI is not a Python command):

```bash
curl -H "X-Requester-Id: admin-cocody" http://127.0.0.1:8000/establishments/lycee-cocody/classes
```

> **⚠️ Authentication is a stub and is not secure.** The requester is whoever the `X-Requester-Id` header says, with no password or token: anyone who can reach the API can act as any teacher or admin. Authorization from the registry is real; identity is not. See `apu/auth/identity.py` before deploying anything.

### Braille

`apu/modality/braille/` translates with liblouis (French BFU or English UEB, grade 1 or grade 2) into Unicode braille or embosser encoding (Braille ASCII / BRF), and includes a simulated embosser. liblouis is a C library installed by the system, not by uv:

```bash
brew install liblouis
```

On Debian/Ubuntu: `apt install liblouis20`. Without it, the braille tests are skipped.

## Repository layout

```
apu/
  config.py                # env loading, model IDs, paths, tunables
  logger.py                # shared logging (console + apu_runtime.log)
  inference/
    nebius_client.py       # Nebius Token Factory client, main + extraction call wrappers
  runtime/
    agent.py               # LangGraph planner: retrieval, prompt, Nemotron calls, memory write-back
  mmu/
    dll.py                 # doubly linked list of memory blocks, LRU paging, BMJ routing
    cache_l1.py            # L1 in-process cache with per-type TTL
  storage/
    lance_driver.py        # L3 LanceDB vector store, embedding-space guard
  core/
    scheduler.py           # background task scheduler (ported stub, see HACKATHON.md)
    extraction.py          # tolerant parser for the extraction model's JSON
    block_detector.py      # proposes new dynamic memory blocks
    block_proposal.py      # the proposal contract between detector and executor
  embeddings/
    local_embedder.py      # local fastembed/ONNX wrapper, swappable per target device
  guardrails/
    config/                # shared NeMo Guardrails config: Nemotron on Token Factory, topical rail (Colang)
    guard.py               # runs the input rail on each turn, fails closed
    actions.py             # classifier, off-topic counting and escalation, turn validation
    policy.py session.py   # per-class policy (read once per session), in-memory session state
  escalation/
    models.py              # EscalationEvent, EscalationResolution, cluster snapshots (immutable)
    clustering.py jobs.py  # HDBSCAN per class, deferred write and recompute jobs
  mmu/
    escalation_store.py    # escalation storage, kept out of the DLL and L3
    block_types.py         # block types the tutoring memory refuses
  auth/
    assignments.py         # registry: who teaches or administers what (only source of roles)
    authorization.py       # authorize_view
    identity.py            # AUTHENTICATION STUB, not secure
  api/
    app.py                 # FastAPI teacher/admin routes
  tools/
    web_search.py          # Tavily search, only for guard-validated turns
  modality/
    mode.py citations.py   # interaction modes, source citations per output channel
    braille/               # liblouis translator, embosser simulator
  sync/
    sync_manager.py        # device side of the cloud registry: catalog, course download, prompts
  ui/
    dashboard.py           # Streamlit APU Control Center (uv run streamlit run apu/ui/dashboard.py)
cloud_registry/            # publishing side of the course registry (see its README)
  config/                  # curriculum.yaml, publishing settings
  courses/                 # Markdown chapters per grade/subject, tutor prompts
  pipeline/                # batch_pipeline.py: embed, build Parquet + manifest, upload to GCS
scripts/
  fetch_embedding_model.py # one-time download of the ONNX embedding model
  smoke_test_tool_calling.py # checks native tool calling on Token Factory (see HACKATHON.md)
registries/                # demo class policies and teacher assignments, loaded at startup
tests/                     # pytest suite, offline, no API key
docs/
  architecture.md          # links to the fuller write-up of the underlying design
```

## Status

Early scaffold, ported from the existing APU / Akili codebase. See [HACKATHON.md](./HACKATHON.md) for the current state and what is left to port.

## License

Apache License 2.0, see [LICENSE](./LICENSE).
