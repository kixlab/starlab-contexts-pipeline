import type { TreeEdge } from "./treeBuilder";

/** Parents + children of selected nodes (excluding the selected nodes themselves). */
export function computeHighlightedNodeIds(
  selectedNodeIds: string[],
  edges: TreeEdge[],
): Set<string> {
  const set = new Set<string>();
  for (const nodeId of selectedNodeIds) {
    for (const e of edges) {
      if (e.sourceId === nodeId) set.add(e.targetId);
      if (e.targetId === nodeId) set.add(e.sourceId);
    }
  }
  for (const nodeId of selectedNodeIds) set.delete(nodeId);
  return set;
}

/** Edges whose source or target is a selected node. Empty when nothing selected. */
export function computeVisibleEdges(
  selectedNodeIds: string[],
  edges: TreeEdge[],
): TreeEdge[] {
  if (selectedNodeIds.length === 0) return [];
  const sel = new Set(selectedNodeIds);
  return edges.filter((e) => sel.has(e.sourceId) || sel.has(e.targetId));
}
