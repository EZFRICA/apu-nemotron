# What existed before this hackathon, and what is new

The hackathon rules require a written explanation of what was significantly updated during the submission period, for any project that existed before it. This file is that explanation.

## Pre-existing work

- The Agent Processor Unit (APU) architecture: four-tier memory hierarchy (L1 in-process cache, L2 Redis working-context list, L3 Weaviate, L4 Letta), BMJ routing, async priority-queue scheduler. Open-sourced at `github.com/EZFRICA/Agent-Processor-Unit`.
- Akili, the education-focused implementation of the APU, targeting offline classroom use on constrained hardware. Akili differs from the reference APU: L1 in-process cache, L2 DLL persisted as JSON, L3 local LanceDB, no L4, no Redis. **This repository is ported from Akili**, not from the reference APU.
- A Medium article series on the Google Cloud Community collection documenting the architecture and its measured behavior on several inference backends (Gemma via Ollama, Gemini variants).

Neither of these used Nebius infrastructure or an NVIDIA open source model before this hackathon.

## New during the submission period

- [x] Ported the main answer call from the Akili provider layer (Ollama / Gemma / Gemini / OpenRouter) to `nvidia/nemotron-3-super-120b-a12b` via Nebius Token Factory (`apu/runtime/agent.py` → `call_main_model`). Verified against a mocked client; **not yet run against the live endpoint**.
- [x] Ported the memory-extraction call to `nvidia/nemotron-3-nano-30b-a3b` via Nebius Token Factory (`call_extraction_model`, temperature 0, JSON mode). Verified against a mocked client; **not yet run against the live endpoint**, and `response_format={"type": "json_object"}` is unverified for Nemotron Nano on Token Factory.
- [ ] Re-ran the turn-latency and retrieval-certainty benchmarks against the Nemotron backends, following the same protocol used for the earlier Gemma / Gemini comparisons. Needs Akili's `scripts/bench_latency.py`, not ported yet.
- [x] Documented the local-embedder-stays-local decision explicitly (README "Why", `apu/embeddings/local_embedder.py`).
- [x] Ported the Akili core into the new layout, behaviour unchanged: DLL/MMU with BMJ routing and LRU page-out (`apu/mmu/dll.py`, merging Akili's `mmu/controller.py` + `mmu/block_factory.py`), L1 cache (`apu/mmu/cache_l1.py`), L3 LanceDB driver (`apu/storage/lance_driver.py`), scheduler (`apu/core/scheduler.py`), extraction parser, block detector and proposal contract (`apu/core/`), local ONNX embedder (`apu/embeddings/local_embedder.py`), LangGraph runtime (`apu/runtime/agent.py`), model fetch script (`scripts/fetch_embedding_model.py`).
- [x] Ported the Akili test suite with the LLM mocks replaced by a fake Nebius client that checks model routing: 164 tests passing offline, real ONNX embedder included when the model cache is present.
- [x] Fixed the default embedding model id: the scaffold's `paraphrase-multilingual-MiniLM-L12-v2` is rejected by fastembed; now `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- [x] Removed `redis` and `weaviate-client` from `requirements.txt` (nothing in the Akili code uses them); pinned `fastembed` and `lancedb` to the tested minors.

## Still to do

- [ ] Run a real turn against Token Factory with a `NEBIUS_API_KEY`, and check the extraction call accepts JSON mode.
- [ ] Decide the block cap: the scaffold documents 12 active blocks (reference APU: 4 fixed + 8 dynamic); Akili, and so this port, uses 4 fixed + 5 dynamic. `MAX_ACTIVE_BLOCKS` exists in `apu/config.py` but is not wired.
- [ ] Decide whether the extraction call moves off the critical path. It still runs inline, awaited before the answer returns, as in Akili. Akili's `COLD_PATH.md` measured −34.6% perceived latency for moving it to a background thread, with known costs.
- [ ] Scheduler: the ported one is Akili's stub (FIFO, priority ignored, no retry, GC log-only, never started). Deferred L4 writes and retry-with-backoff exist only in the reference APU.
- [ ] Page-fault / page-in of a paged-out block back into the DLL: reference APU only, not in Akili.
- [ ] Tool Execution Unit (calculator, course search, chapter loader): not ported, because `call_main_model` returns only the text and drops `tool_calls`.
- [ ] Course content: the sync manager that downloads the course registry (`edu_registry`) and `prompts.json` is not ported, so a fresh install has student memory but no curriculum.
- [ ] `delete_block_stitching` calls `lance_driver.delete_local_block`, which does not exist (already true in Akili); pinned by a test.
- [ ] Not ported: Streamlit dashboard, cloud registry pipeline, `scripts/migrate_embeddings.py`, `scripts/dedup_user_memory.py`, `scripts/bench_latency.py`, CI workflow.
- [ ] README layout and tier description (Redis L2, Weaviate L3, Letta L4) describe the reference APU, not this port.
