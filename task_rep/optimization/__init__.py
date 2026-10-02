"""
Schema optimization module.

Provides schema metrics, inter-rater reliability, and refinement tree
construction for contextual-unit pool evaluation.

Public API:
- compute_cells, compute_discriminativeness, compute_compactness, ...: Schema metrics
- compute_irr: Inter-rater reliability for facet labeling
- TreeNode, TreeEdge, RefinementTree, build_tree_nodes, evaluate_schemas, ...: Refinement tree
"""

from .metrics import (
    compute_cells,
    compute_discriminativeness,
    compute_compactness,
    compute_eta_squared,
    compute_quality_metric,
    count_unique_units,
)
from .irr import (
    compute_irr,
)
from .tree import (
    TreeNode,
    TreeEdge,
    RefinementTree,
    AcceptedSchemaSummary,
    RejectedSchemaSummary,
    SchemaEvaluation,
    build_tree_nodes,
    prune_and_connect,
    evaluate_schemas,
    compare_tree_quality,
    compute_tree_depth,
)
