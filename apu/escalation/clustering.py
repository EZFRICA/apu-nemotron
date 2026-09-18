"""Clustering of a class's escalation events, so a teacher sees patterns rather than a list.

Scoped to one class, always: events from other classes are dropped before clustering, and
nothing here aggregates at establishment level.

Embeddings come from the existing local embedder (no new embedding dependency). HDBSCAN
rather than a greedy similarity threshold: the result does not depend on the order events
arrived in, and its noise label (-1) leaves an isolated request outside every cluster
instead of forcing it into the nearest one.

Computed by a periodic job (apu.escalation.jobs), never at read time.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import HDBSCAN

from apu.embeddings import local_embedder
from apu.escalation.models import EscalationCluster, EscalationClusterSnapshot, EscalationEvent

MIN_CLUSTER_SIZE = 2
NOISE_LABEL = -1

EmbedTexts = Callable[[list[str]], Sequence[Sequence[float]]]


def _default_embed_texts(texts: list[str]) -> Sequence[Sequence[float]]:
    return local_embedder.get_embedder().embed_documents(texts)


def cluster_class_events(
    class_id: str,
    events: Sequence[EscalationEvent],
    embed_texts: EmbedTexts | None = None,
    now: datetime | None = None,
) -> EscalationClusterSnapshot:
    computed_at = now or datetime.now(UTC)
    class_events = sorted(
        (event for event in events if event.class_id == class_id),
        key=lambda event: (event.triggered_at, event.event_id),
    )
    # HDBSCAN refuses a single sample, and fewer events than the minimum cluster size can
    # never form a cluster anyway.
    if len(class_events) < MIN_CLUSTER_SIZE:
        return EscalationClusterSnapshot(class_id=class_id, computed_at=computed_at, clusters=[])

    embed = embed_texts or _default_embed_texts
    vectors = np.asarray(
        [local_embedder.normalize_vector(vector)
         for vector in embed([event.off_topic_request_text for event in class_events])],
        dtype=float,
    )
    labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="cosine").fit(vectors).labels_

    members_by_label: dict[int, list[EscalationEvent]] = {}
    # strict: one label per event by construction; a mismatch would silently drop events.
    for event, label in zip(class_events, labels, strict=True):
        if int(label) == NOISE_LABEL:
            continue
        members_by_label.setdefault(int(label), []).append(event)

    # Stable ids across recomputations of the same data: clusters numbered by their
    # earliest event, not by HDBSCAN's internal label order.
    ordered_groups = sorted(members_by_label.values(), key=lambda members: members[0].triggered_at)
    clusters = [
        EscalationCluster(
            cluster_id=index,
            event_ids=[event.event_id for event in members],
            representative_text=members[0].off_topic_request_text,
        )
        for index, members in enumerate(ordered_groups)
    ]
    return EscalationClusterSnapshot(class_id=class_id, computed_at=computed_at, clusters=clusters)
