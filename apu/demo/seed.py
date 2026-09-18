"""Live-demo preparation: reset per-device state, load courses locally, seed escalations and a notebook.

Everything here writes only to the application's own state paths (and to
cloud_registry/registry/ for the locally built course registry), never to a bucket and
never to the tutoring memory with escalation data.

  - Courses: the cloud pipeline is run locally (same embedding, same Parquet files as a
    published registry) and each course is imported through the same function the
    device uses after a bucket download. No Google credentials needed.
  - Escalations: example events for two demo classes, written straight to the escalation
    store (this is data preparation, not a student turn), then clustered by the same
    job function the deferred scheduler runs.

The example requests use close phrasings on purpose. Measured with the local MiniLM
embedder: different phrasings of one theme ("the score of the CAN final" vs "who won the
PSG match") only reach 0.26-0.60 cosine similarity and form no cluster with the specified
HDBSCAN parameters, and an isolated request next to tight groups can be absorbed into one
of them. See HACKATHON.md.
"""

import asyncio
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from apu import config
from apu.escalation.jobs import recompute_class_snapshot
from apu.escalation.models import EscalationEvent
from apu.guardrails.session import sessions
from apu.mmu import cache_l1
from apu.mmu import dll as mmu
from apu.mmu.escalation_store import EscalationStore
from apu.notebook.store import EntryOrigin, NotebookStore, new_entry
from apu.storage import lance_driver

# (class_id, student_id, request, minutes before now). Two tight themes per class.
DEMO_ESCALATIONS: list[tuple[str, str, str, int]] = [
    ("lycee-cocody:3eA", "eleve-koffi", "Who won the PSG vs Marseille match last night?", 190),
    ("lycee-cocody:3eA", "eleve-ibrahim", "What was the score of the PSG vs Marseille match?", 170),
    ("lycee-cocody:3eA", "eleve-awa", "Did PSG beat Marseille last night or not?", 150),
    ("lycee-cocody:3eA", "eleve-junior", "How do I get free diamonds in Free Fire?", 120),
    ("lycee-cocody:3eA", "eleve-fatou", "Tips to get free diamonds in Free Fire", 95),
    ("lycee-cocody:3eA", "eleve-serge", "Give me a code for free diamonds in Free Fire", 60),
    ("lycee-cocody:4eB", "eleve-mariam", "What is the best Didi B song?", 80),
    ("lycee-cocody:4eB", "eleve-ange", "Which Didi B song is the best right now?", 40),
    ("lycee-cocody:4eB", "eleve-paul", "How do I make a TikTok video go viral?", 30),
    ("lycee-cocody:4eB", "eleve-nina", "Tips to make a TikTok video go viral", 15),
]

# (class_level, subject, kind, text, minutes before now): what the demo student has already
# kept, so the notebook and its revision sheet have something to show from the first minute.
DEMO_NOTEBOOK: list[tuple[str, str, str, str, int]] = [
    ("6eme", "math", "key_points",
     "- A fraction a/b means a parts out of b equal parts.\n"
     "- The denominator b says how many equal parts the whole is cut into.\n"
     "- The numerator a says how many of those parts are taken.", 2 * 24 * 60),
    ("6eme", "math", "excerpt",
     "To add two fractions, first put them over the same denominator, then add the numerators "
     "and keep the denominator: 1/4 + 1/6 = 3/12 + 2/12 = 5/12.", 24 * 60),
    ("6eme", "math", "full",
     "Two fractions are equivalent when they name the same share of a whole. Multiply or divide "
     "the numerator and the denominator by the same number: 2/3 = 4/6 = 8/12.", 90),
]

Progress = Callable[[str], None]


def _state_paths() -> list[str]:
    """
    The exact paths the application writes per-device state to.

    Listed one by one from the configuration the code actually reads, rather than
    "everything under DATA_DIR": a reset must never remove something that is not this
    application's state, even if the data directory is shared or redirected.
    """
    from apu.sync import sync_manager

    data_parent = os.path.dirname(config.LANCE_DB_PATH)
    return [
        config.LANCE_DB_PATH,
        os.path.dirname(config.METADATA_LINKS_PATH),
        config.CACHE_DIR,
        config.ESCALATION_DB_PATH,
        config.NOTEBOOK_DB_PATH,
        config.EMBEDDING_STAMP_PATH,
        sync_manager.LOCAL_MANIFEST_PATH,
        os.path.join(data_parent, "prompts.json"),
    ]


@dataclass
class DemoReport:
    courses: dict[str, int] = field(default_factory=dict)   # "6eme/math" -> chapters imported
    prompts_loaded: bool = False
    escalations_by_class: dict[str, int] = field(default_factory=dict)
    clusters_by_class: dict[str, int] = field(default_factory=dict)
    notebook_entries: int = 0


def reset_demo_data(progress: Progress = print) -> None:
    """Remove the device's state (memory, courses, escalations) and every open guard session."""
    for path in _state_paths():
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    # Connections and caches still point at the deleted files otherwise.
    lance_driver._db = None
    cache_l1.flush_all()
    sessions.clear()
    progress("Local data reset (memory, notebooks, courses, escalations).")


def load_local_course_registry(progress: Progress = print) -> tuple[dict[str, int], bool]:
    """Build the course registry locally with the cloud pipeline and import it."""
    import yaml

    from apu.sync import sync_manager
    from cloud_registry.pipeline import batch_pipeline

    with open(batch_pipeline.CURRICULUM_PATH, encoding="utf-8") as curriculum_file:
        curriculum = yaml.safe_load(curriculum_file)

    imported: dict[str, int] = {}
    for class_level, class_data in curriculum.get("classes", {}).items():
        for subject in class_data.get("subjects", []):
            progress(f"Embedding course {class_level}/{subject}…")
            chapters = asyncio.run(batch_pipeline.process_one(class_level, subject))
            if not chapters:
                continue
            parquet = batch_pipeline.REGISTRY_DIR / f"{class_level}_{subject}_v1.parquet"
            imported[f"{class_level}/{subject}"] = sync_manager.import_course_parquet(
                str(parquet), class_level, subject, file_hash="local-demo-build"
            )

    prompts_loaded = False
    prompts_source = batch_pipeline.export_prompts()
    if prompts_source:
        # Where the registry sync writes prompts, and where the agent reads them.
        target = os.path.join(os.path.dirname(config.LANCE_DB_PATH), "prompts.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(prompts_source, target)
        prompts_loaded = True
    progress(f"{sum(imported.values())} chapters imported, prompts {'loaded' if prompts_loaded else 'missing'}.")
    return imported, prompts_loaded


def seed_escalations(
    store: EscalationStore | None = None,
    embed_texts=None,
    now: datetime | None = None,
    progress: Progress = print,
) -> tuple[dict[str, int], dict[str, int]]:
    from apu.guardrails.policy import get_class_policy_registry

    store = store or EscalationStore()
    now = now or datetime.now(UTC)
    policies = get_class_policy_registry()
    thresholds: dict[str, int] = {}

    counts: dict[str, int] = {}
    for class_id, student_id, request, minutes_ago in DEMO_ESCALATIONS:
        threshold = thresholds.setdefault(class_id, policies.get(class_id).escalation_threshold)
        store.append_event(EscalationEvent(
            event_id=str(uuid.uuid4()),
            student_id=student_id,
            class_id=class_id,
            session_id=f"demo-{student_id}",
            attempt_number_in_session=threshold,
            off_topic_request_text=request,
            triggered_at=now - timedelta(minutes=minutes_ago),
        ))
        counts[class_id] = counts.get(class_id, 0) + 1

    clusters: dict[str, int] = {}
    for class_id in counts:
        recompute_class_snapshot(class_id, store, embed_texts)
        snapshot = store.latest_snapshot(class_id)
        clusters[class_id] = len(snapshot.clusters) if snapshot else 0
    progress("Example escalations: " + ", ".join(
        f"{class_id} → {counts[class_id]} events, {clusters[class_id]} clusters" for class_id in counts
    ))
    return counts, clusters


def seed_notebook(
    student_id: str = config.DEMO_STUDENT_ID,
    store: NotebookStore | None = None,
    now: datetime | None = None,
    progress: Progress = print,
) -> int:
    store = store or NotebookStore()
    now = now or datetime.now(UTC)
    for class_level, subject, kind, text, minutes_ago in DEMO_NOTEBOOK:
        store.add(new_entry(student_id, class_level, subject, kind, text, source_answer=text,
                            origin=EntryOrigin.BUTTON, created_at=now - timedelta(minutes=minutes_ago)))
    progress(f"Notebook of {student_id}: {len(DEMO_NOTEBOOK)} saved entries.")
    return len(DEMO_NOTEBOOK)


def prepare_demo(*, reset: bool = True, load_courses: bool = True, progress: Progress = print) -> DemoReport:
    report = DemoReport()
    if reset:
        reset_demo_data(progress)
    if load_courses:
        report.courses, report.prompts_loaded = load_local_course_registry(progress)
    report.escalations_by_class, report.clusters_by_class = seed_escalations(progress=progress)
    report.notebook_entries = seed_notebook(progress=progress)
    asyncio.run(mmu.force_reinit_dll())
    progress("Student memory (DLL) back to its initial state.")
    return report


# ── diagnostics shown on the demo setup page ─────────────────────────────────

@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str


def demo_checks() -> list[Check]:
    checks = [
        Check("Nebius Token Factory key", bool(os.environ.get("NEBIUS_API_KEY")), "NEBIUS_API_KEY in .env"),
        Check("Tavily key", bool(os.environ.get("TAVILY_API_KEY")), "TAVILY_API_KEY in .env"),
    ]
    model_ready = os.path.isdir(config.LOCAL_EMBEDDING_CACHE_DIR) and any(
        "paraphrase-multilingual" in name for name in os.listdir(config.LOCAL_EMBEDDING_CACHE_DIR)
    )
    checks.append(Check("Local embedding model", model_ready, config.LOCAL_EMBEDDING_CACHE_DIR))
    courses = loaded_courses()
    checks.append(Check("Courses loaded", bool(courses), ", ".join(courses) or "none"))
    store = EscalationStore()
    escalations = {
        class_id: len(store.events_for_class(class_id))
        for class_id in {entry[0] for entry in DEMO_ESCALATIONS}
    }
    checks.append(Check(
        "Example escalations", any(escalations.values()),
        ", ".join(f"{class_id}: {count}" for class_id, count in sorted(escalations.items())),
    ))
    return checks


def loaded_courses() -> list[str]:
    try:
        db = lance_driver.get_db()
        if "edu_registry" not in lance_driver.list_table_names(db):
            return []
        pairs = db.open_table("edu_registry").to_pandas()[["class_level", "subject"]].drop_duplicates()
    except Exception:
        return []
    return sorted(f"{c}/{s}" for c, s in pairs.itertuples(index=False))


def live_check_nemotron() -> str:
    from apu.inference import nebius_client
    reply = nebius_client.call_main_model(
        [{"role": "user", "content": "Reply with just: OK"}], temperature=0.0
    )
    return (reply or "").strip()


def live_check_tavily() -> int:
    from langchain_tavily import TavilySearch as LangchainTavilySearch
    result = LangchainTavilySearch(max_results=1).invoke({"query": "BEPC exam dates Côte d'Ivoire"})
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(str(result["error"]))
    return len(result.get("results", [])) if isinstance(result, dict) else 0


def load_demo_students(path: str | None = None) -> list[dict]:
    with open(path or config.DEMO_STUDENTS_PATH, encoding="utf-8") as students_file:
        return json.load(students_file)
