"""
Clustering utilities for knowledge items.

Clusters knowledge items by semantic similarity using embeddings.
Items of different types are never clustered together.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List
from collections import defaultdict

from utils import (
    ModelConfig,
    get_embeddings,
)

from .schema import KnowledgeItem


@dataclass
class ClusterConfig:
    """Configuration for clustering knowledge items."""
    
    model_config: ModelConfig
    similarity_threshold: float = 0.85

    def cache_suffix(self) -> str:
        """Generate cache suffix for this config."""
        threshold_str = str(self.similarity_threshold).replace(".", "")
        model_name = self.model_config.cache_suffix()
        return f"sim{threshold_str}_{model_name}"


def _cluster_greedy(
    embeddings: np.ndarray,
    similarity_threshold: float
) -> List[int]:
    """
    Greedy clustering based on cosine similarity.

    For each unvisited item, cluster it with all similar unvisited items.
    Returns list of cluster labels.
    """
    n = len(embeddings)

    if n <= 1:
        return [0] * n

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    # Compute similarity matrix
    similarities = normalized @ normalized.T

    # Greedy clustering
    labels = list(range(n))
    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        for j in range(i + 1, n):
            if visited[j]:
                continue
            if similarities[i, j] >= similarity_threshold:
                visited[j] = True
                labels[j] = labels[i]

    return labels


def cluster_knowledge_items(
    items: List[KnowledgeItem],
    config: ClusterConfig,
) -> List[KnowledgeItem]:
    """
    Cluster knowledge items by semantic similarity.

    Items of different types are never clustered together.

    Args:
        items: List of knowledge items to cluster
        config: Clustering configuration (threshold, model config)

    Returns:
        New list of items with unit_id assigned
    """
    if not items:
        return []

    # Group by type - items of different types never cluster together
    items_by_type = defaultdict(list)
    for idx, item in enumerate(items):
        items_by_type[item.type].append(idx)

    # Cluster each type separately
    unit_ids = [None] * len(items)
    global_unit_id = 0

    for _, indices in items_by_type.items():
        if not indices:
            continue

        # Extract texts and get embeddings
        texts = [items[idx].text for idx in indices]
        embeddings = get_embeddings(texts, model_config=config.model_config)

        # Cluster
        local_labels = _cluster_greedy(embeddings, config.similarity_threshold)

        # Map local cluster IDs to global unit IDs
        local_to_global = {}
        for local_idx, local_label in enumerate(local_labels):
            if local_label not in local_to_global:
                local_to_global[local_label] = f"unit_{global_unit_id}"
                global_unit_id += 1

            global_idx = indices[local_idx]
            unit_ids[global_idx] = local_to_global[local_label]

    # Create new items with unit_id assigned
    clustered_items = []
    for item, unit_id in zip(items, unit_ids):
        clustered_items.append(
            KnowledgeItem(
                id=item.id,
                type=item.type,
                text=item.text,
                source=item.source,
                subtype=item.subtype,
                unit_id=unit_id,
                metadata=item.metadata,
            )
        )

    return clustered_items
