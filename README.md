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

```bash
git clone <this-repo-url>
cd apu-nemotron
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in NEBIUS_API_KEY in .env, get one at https://tokenfactory.nebius.com/
```

## Repository layout

```
apu/
  config.py              # env loading, model IDs, tunables
  inference/
    nebius_client.py     # Nebius Token Factory client, main + extraction call wrappers
  core/
    scheduler.py          # async priority-queue scheduler, deferred writes, background GC
  mmu/
    dll.py                 # doubly linked list of memory blocks, LRU paging, BMJ routing
  embeddings/
    local_embedder.py     # local fastembed/ONNX wrapper, swappable per target device
tests/
docs/
  architecture.md          # links to the fuller write-up of the underlying design
```

## Status

Early scaffold, ported from the existing APU / Akili codebase. See [HACKATHON.md](./HACKATHON.md) for the current state and what is left to port.

## License

Apache License 2.0, see [LICENSE](./LICENSE).
