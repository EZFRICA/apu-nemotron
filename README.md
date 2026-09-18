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
| `APU_STUDENT_ID` / `APU_CLASS_ID` | `eleve-aya` / `lycee-cocody:3eA` | Identity the interface opens with; the sidebar switches it (stub, no login) |
| `APU_ESCALATION_CLUSTER_TRIGGER_COUNT` | `5` | New events in a class before its clusters are recomputed in the background |
| `APU_DEMO_STUDENTS_PATH` | `registries/demo_students.json` | Demo students offered by the identity selector (stub) |
| `TAVILY_MAX_RESULTS` | `5` | Results kept per web search |

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

or set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to a service-account JSON with read access to the bucket. Then, in the interface (step 7), open **📚 Active course** on the student page, click **Browse the cloud registry**, pick a course and click **⬇️ Download and activate**. **🔄 Check for updates (prompts)** refreshes the system prompts from the registry. Courses stay usable offline once downloaded.

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
uv run streamlit run apu/ui/app.py
```

Streamlit prints a local URL, `http://localhost:8501` by default: open it in a browser. The project's `.streamlit/config.toml` runs it headless (no browser auto-open, no first-run prompt, no usage statistics) with a dark theme.

What you get, in three pages:

- **Student.** The tutor chat. Every question first goes through the topical guard (Nemotron Super), then gets its answer (Nemotron Super, with Tavily web search when the model needs it) and the memory extraction (Nemotron Nano), all before the answer is displayed. Above the chat: the off-topic counter against the class threshold, and the guard's last verdict. Under each answer, **💾 Save to notebook** keeps it in the student's notebook. Tabs show the notebook and its revision sheets, the last turn's searches and sources, and the memory hierarchy (L1 cache, L2 DLL chain, L3 LanceDB tables). The **📚 Active course** panel activates a local course, downloads one from the cloud registry (**Browse the cloud registry**, **⬇️ Download and activate**), refreshes prompts, and resets the student's memory.
- **Teacher / Admin.** A class's escalations (resolve with a note), their clusters, and an access-control check that shows a refusal outside the person's scope.
- **Demo setup.** Environment checks, live tests of Nemotron and Tavily, and demo data preparation.

The sidebar's **Sign in as** switches between demo students, teachers and admins. It is a stub, not authentication: see [Live demo](#live-demo).

To use another port:

```bash
uv run streamlit run apu/ui/app.py --server.port 8502
```

### 8. Where state lives, and how to reset it

Everything the device holds is under `data/` (gitignored): `data/memory/metadata_links.json` is the DLL (L2), `data/akili_db/` the LanceDB store (L3: downloaded courses in `edu_registry`, archived student memory in `user_memory`), `data/local_manifest.json` the list of downloaded courses, `data/prompts.json` the system prompts from the registry, `data/escalations.sqlite3` the escalation events, `data/notebook.sqlite3` the student notebooks. **🗑️ Reset the student's memory (L1 + L2)** (student page) wipes L1 and L2; rows already archived in L3 stay. To reset everything and reload demo data, use the demo preparation (below), or stop Streamlit and delete the directory:

```bash
rm -rf data
```

The runtime log is `data/apu_runtime.log`, at INFO. It is deliberately not DEBUG: this file
outlives the process and the users are minors, and DEBUG carries block contents.

### 9. Student data: what is kept, what leaves the device

| What | Where | Who reads it | How to erase it |
| --- | --- | --- | --- |
| Questions and answers of the current conversation | In memory only, in the browser session | The student | Reload the page |
| Profile, preferences, current session (extracted by Nemotron Nano) | `data/memory/`, `data/akili_db/` (this device) | The tutor, on the next turn | **🗑️ Reset the student's memory** on the student page |
| Notebook entries | `data/notebook.sqlite3` | The student only; never the tutor | 🗑️ on each entry, or `scripts/purge_data.py --student <id>` |
| Escalation events (the off-topic message, the pupil id, the moment) | `data/escalations.sqlite3` | The class's teacher and the school admin, through the API's authorization | `scripts/purge_data.py` |

Two things leave the device: **every question goes to Nebius Token Factory** (the guard's
classification, the answer, the memory extraction), and **a web search sends its query to
Tavily**. The query is written by the model from the question; the tutor is instructed never
to put the student's memory into it, and the interface shows every query it sent. Nothing
else is uploaded: the notebook, the escalations and the memory stay on the device.

Nothing expires on its own. A school running this owes itself a retention period; this is
what runs it, from cron or by hand:

```bash
uv run python scripts/purge_data.py --older-than-days 180
```

### Troubleshooting

- **`⚠️ This turn could not be completed: NEBIUS_API_KEY is not set` in the chat.** Set the key in `.env`, then stop Streamlit (Ctrl+C) and launch it again. The **Demo setup** page checks every key and dependency at a glance.
- **`Embedding model ... is not present in ...`.** Step 4 was not run, or `LOCAL_EMBEDDING_CACHE_DIR` points to the wrong place.
- **"Registry unreachable" in the sidebar.** The device could not read the registry bucket: no Google credentials (step 5), no network, or a wrong `REGISTRY_MANIFEST_URL`. The terminal running Streamlit prints the exact reason (`[Sync] Auth failed: ...`). Courses already downloaded remain available.
- **"This registry was built with '...'" when downloading.** The registry and this device use different embedding models; align `LOCAL_EMBEDDING_MODEL` with the registry's, or republish the registry.
- **`Port 8501 is already in use`.** Use `--server.port` as shown above.
- **`⚠️ This turn could not be completed: The topical guard could not classify this turn`.** Every question first goes through the topical guard, which calls Nemotron on Token Factory. Same fix as a missing or invalid `NEBIUS_API_KEY`. The tutor deliberately does not answer when the guard cannot run.

## Live demo

The full run sheet is in [DEMO.md](./DEMO.md). In short:

1. With `.env` holding `NEBIUS_API_KEY` and `TAVILY_API_KEY`, prepare the demo data. This resets the local state under `data/`, builds and imports the courses locally (no Google credentials), and seeds example escalations with their clusters:
   ```bash
   uv run python scripts/prepare_demo.py
   ```
2. Launch the interface and open the printed URL:
   ```bash
   uv run streamlit run apu/ui/app.py
   ```
3. Check **Demo setup**: every line should be ✅, and the two live tests should answer.

> **Identity is simulated.** The sidebar lets anyone act as any demo student, teacher or admin, with no password. What the chosen person can see and do is real: it comes from `registries/` through the same service functions as the API.

## Guardrails, escalations and the teacher API

### Topical guard (NeMo Guardrails)

Every student turn goes through one shared input rail before anything else (`apu/guardrails/config/`, Colang). A classifier running on Nemotron Super through Token Factory decides **strictly school use** (lessons, exercises, revision, study-related research) versus **off-topic**. There is no list of allowed subjects.

What varies per class is data, not Colang: `registries/class_policies.json` holds each class's `escalation_threshold` and extra search exclusions. A session reads its class policy **once, when it opens**; a changed threshold applies to the next session.

- Off-topic, below the threshold: a kind redirect towards schoolwork.
- Off-topic, reaching the threshold: a firmer reply, and one `EscalationEvent` written in the background. Further attempts in the same session stay firm without adding events.
- The off-topic counter lives in memory for the session and restarts on reconnection.
- If the classifier cannot be reached, the turn is not answered.

### Web search (Tavily)

`apu/tools/web_search.py` only searches for a turn the topical guard validated, excludes social networks for everyone (`GLOBAL_EXCLUDED_DOMAINS`) plus each class's own additions, and returns its sources. `apu/modality/citations.py` renders them per output channel: a list at the end in text or braille; source names said aloud (never URLs) in voice, plus the written list when a screen is available. Nemotron requests a search through native OpenAI tool calls (Method A, chosen from the tool-calling smoke test, see [HACKATHON.md](./HACKATHON.md)). The `web_search` tool is only offered on turns the guard validated, and a turn makes at most 2 searches before the model must answer.

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

### Student notebook

During a conversation the student keeps what matters to them in a notebook (`apu/notebook/`), and revision sheets are made from it. Each save keeps one of three things, the student's choice: the **full answer**, its **key points** (condensed by Nemotron Nano, using only what the answer says), or an **excerpt** the student picks.

- **Two ways to save.** The **💾 Save to notebook** control under each answer, or by asking the tutor ("save the key points", "just keep the rule for adding fractions"): Nemotron Super calls the `save_to_notebook` tool. The tool is offered only on turns the guard validated, and the entry is filed under the student of the guard session, never a student named by the model.
- **Revision sheets.** In the **📓 Notebook** tab, pick entries, then generate a sheet from them as written, or from a revision summary Nemotron Super writes from them, and download it.
- **The tutor never reads the notebook.** No entry is ever put into a prompt, and the store is its own SQLite file, apart from the DLL and L3. A test checks that saved text never reaches a model call.

## Repository layout

```
apu/
  config.py                # env loading, model IDs, paths, tunables
  logger.py                # shared logging (console + apu_runtime.log)
  inference/
    nebius_client.py       # Nebius Token Factory client, main + extraction call wrappers
  runtime/
    agent.py               # LangGraph planner: retrieval, prompt, Nemotron calls, memory write-back
    prompts.py             # the registry's system prompts, validated before they are used
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
  notebook/
    store.py               # notebook entries (full answer, key points, excerpt), SQLite
    service.py             # saving, key points (Nano), revision summary (Super)
  tools/
    web_search.py          # Tavily search, only for guard-validated turns
    notebook.py            # save_to_notebook tool, only for guard-validated turns
  modality/
    citations.py           # web search sources appended to an answer
    plain_text.py          # strips Markdown and math delimiters from notebook entries
  sync/
    sync_manager.py        # device side of the cloud registry: catalog, course download, prompts
  ui/
    app.py                 # Streamlit demo interface (uv run streamlit run apu/ui/app.py)
    common.py              # identity selector (stub), guard session, rendering helpers
    views/                 # student.py, teacher.py, demo.py
  demo/
    seed.py                # demo data preparation, reset, environment checks
cloud_registry/            # publishing side of the course registry (see its README)
  config/                  # curriculum.yaml, publishing settings
  courses/                 # Markdown chapters per grade/subject, tutor prompts
  pipeline/                # batch_pipeline.py: embed, build Parquet + manifest, upload to GCS
scripts/
  fetch_embedding_model.py # one-time download of the ONNX embedding model
  smoke_test_tool_calling.py # checks native tool calling on Token Factory (see HACKATHON.md)
  prepare_demo.py          # resets local state, loads courses locally, seeds example escalations
.streamlit/config.toml     # headless, no telemetry, dark theme
registries/                # demo class policies, teacher assignments and demo students
DEMO.md                    # live demo run sheet
tests/                     # pytest suite, offline, no API key
docs/
  architecture.md          # links to the fuller write-up of the underlying design
```

## Known limits

Stated rather than hidden. None of these is a surprise; each is a decision with its reason.

- **The tutoring memory belongs to the device, not to the student.** The DLL, the L1 cache
  and the archived rows are keyed per device, as in the original design (one tablet, one
  pupil). The notebook and the escalations, added here, are per student. Because the demo's
  sidebar switches pupils on one machine, the interface wipes L1 and L2 whenever the person
  changes. Serving several pupils from one process would need the memory keyed by student
  first.
- **There is no authentication.** The interface hands out the teacher and admin views to
  whoever opens the page, and the API believes the `X-Requester-Id` header. Authorization is
  real (it comes from the registry, never from the request); identity is not. The interface
  therefore binds to localhost only (`.streamlit/config.toml`).
- **One student at a time.** Storage calls are `async def` but do blocking I/O, so several
  concurrent students would serialise on the event loop. Model calls do run in worker
  threads. This is fine for one tablet and would need revisiting behind a shared server.
- **No rate limiting.** Nothing bounds how many turns a student takes, and each one costs at
  least two model calls. The notebook is capped (entries per student, entries per sheet);
  turns are not.
- **The API is unversioned** while it is pre-1.0 and consumed only by this repository's own
  interface. Listing is paged; give it a `/v1` prefix before anyone else integrates.
- **The registry is trusted infrastructure.** It ships the tutor's system prompt and the
  course content that goes into it. Downloads are checked against the manifest's sha256 and
  prompts are validated before use (shape and size), but a legitimate registry that is
  compromised can still change what the tutor is.
- **Clustering is fragile on small or varied classes** — measured, with numbers, in
  `HACKATHON.md`.

## Status

Early scaffold, ported from the existing APU / Akili codebase. See [HACKATHON.md](./HACKATHON.md) for the current state and what is left to port.

## License

Apache License 2.0, see [LICENSE](./LICENSE).
