# Architecture

The design rationale behind the Agent Processor Unit (why a doubly linked list instead of a
graph, the BMJ routing algorithm, the four-tier memory hierarchy, measured latency and
memory footprint on constrained hardware) is documented in the Medium article series:

- "From Agent OS to Agent Processor Unit"
- "Implementing the Agent Processor Unit: A Tutor That Runs Offline"

## What this port changed

- **Inference** runs on NVIDIA Nemotron through Nebius Token Factory, with two models per
  turn: Super writes the answer, Nano extracts what to remember (`apu/inference/`,
  `apu/runtime/agent.py`). Ollama and Gemma are gone.
- **A topical guard** (NeMo Guardrails, Colang) runs before every turn and fails closed;
  per-class policy is data, not Colang (`apu/guardrails/`).
- **Web search** (Tavily) and **notebook saves** are native tool calls, offered only on a
  turn the guard validated and executed only with that turn's proof (`apu/tools/`).
- **Escalations** live outside the tutoring memory, in their own SQLite store, clustered by
  a background job and read only through the teacher API (`apu/escalation/`, `apu/api/`).
- **The student notebook** is written on the student's request and never read by the tutor
  (`apu/notebook/`).

## Shape of the code, and what it costs

Three decisions run through the whole codebase. They are deliberate, they have a price, and
the price is written here rather than discovered later.

- **Module-level singletons.** The guard, the session registry, the two registries, the
  embedder, the LanceDB connection and the deferred-write scheduler are process globals with
  `set_*` helpers for tests. It suits one device serving one pupil, and it is what stands
  between this code and running several pupils, or tests, in parallel. Introducing an
  application context is the first step towards a shared server.
- **`async def` over blocking I/O.** SQLite, LanceDB and the file writes are synchronous
  inside coroutines; only the model calls and the embedder go through worker threads. On one
  device nothing waits on anything. Behind a shared server, these calls would need the same
  treatment as the model calls.
- **Three error conventions.** The registry sync returns `(ok, message)` tuples (ported from
  Akili and surfaced directly in the interface), the stores raise domain exceptions, and the
  agent returns problem lists in its state so the interface can show what degraded without
  failing the turn. They coexist on purpose: the third one exists because a tutor that
  cannot write memory should still answer the question. New code should raise.

The memory hierarchy itself (L1 cache, L2 DLL, L3 LanceDB), the BMJ routing and the LRU
page-out are unchanged from the source implementation; `HACKATHON.md` records what was
measured, what is still open, and where this port deliberately differs.
