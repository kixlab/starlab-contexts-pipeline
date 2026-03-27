"""
Schema optimization module.

Provides metrics, Pareto frontier computation, refinement tree construction,
and collision detection for iterative task schema discovery.

Public API:
- compute_cells, compute_discriminativeness, compute_compactness: Schema metrics
- FrontierPoint, compute_frontier, clip_frontier, is_pareto_improved, sample_schemas: Pareto frontier
- TreeNode, TreeEdge, RefinementTree, build_tree_nodes, prune_and_connect, ...: Refinement tree
- sample_target_items: Collision detection
"""

from .metrics import (
    compute_cells,
    compute_discriminativeness,
    compute_compactness,
    compute_eta_squared,
    compute_quality_metric,
    count_unique_units,
)
from .frontier import (
    FrontierPoint,
    compute_frontier,
    clip_frontier,
    is_pareto_improved,
    sample_schemas,
)
from .collision import (
    sample_target_items,
    sample_target_items_coherence,
)
from .irr import (
    compute_irr,
)
from .tree import (
    TreeNode,
    TreeEdge,
    RefinementTree,
    build_tree_nodes,
    prune_and_connect,
    sample_schemas_from_tree,
    is_tree_improved,
    get_active_facet_ids,
    compute_tree_depth,
)