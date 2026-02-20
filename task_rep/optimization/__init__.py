"""
Schema optimization module.

Provides metrics, Pareto frontier computation, and collision detection
for iterative task schema discovery.

Public API:
- compute_cells, compute_discriminativeness, compute_compactness: Schema metrics
- FrontierPoint, compute_frontier, clip_frontier, is_pareto_improved, sample_schemas: Pareto frontier
- find_colliding_cells, sample_target_items: Collision detection
"""

from .metrics import compute_cells, compute_discriminativeness, compute_compactness, count_unique_units
from .frontier import FrontierPoint, compute_frontier, clip_frontier, is_pareto_improved, sample_schemas
from .collision import find_colliding_cells, sample_target_items

__all__ = [
    # Metrics
    "compute_cells",
    "compute_discriminativeness",
    "compute_compactness",
    "count_unique_units",
    # Frontier
    "FrontierPoint",
    "compute_frontier",
    "clip_frontier",
    "is_pareto_improved",
    "sample_schemas",
    # Collision
    "find_colliding_cells",
    "sample_target_items",
]
