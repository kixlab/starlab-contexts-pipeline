import type { RefinementTree, TreeEdge } from "./treeBuilder";
import type { VisibleEdge } from "../types";

/** Number of direct children per node (from full tree, stable). */
export function computeChildrenCountMap(
  tree: RefinementTree,
): Map<string, number> {
  const map = new Map<string, number>();
  for (const n of tree.nodes) map.set(n.id, 0);
  for (const e of tree.edges) {
    map.set(e.sourceId, (map.get(e.sourceId) ?? 0) + 1);
  }
  return map;
}

/** BFS from roots — children are revealed only if parent is expanded. */
export function computeVisibleNodeIds(
  tree: RefinementTree,
  expandedNodeIds: Set<string>,
): Set<string> {
  const childrenOf = new Map<string, string[]>();
  for (const e of tree.edges) {
    let arr = childrenOf.get(e.sourceId);
    if (!arr) { arr = []; childrenOf.set(e.sourceId, arr); }
    arr.push(e.targetId);
  }

  const visible = new Set<string>();
  const queue = [...tree.roots];
  for (const id of queue) visible.add(id);

  while (queue.length > 0) {
    const id = queue.shift()!;
    if (!expandedNodeIds.has(id)) continue;
    const kids = childrenOf.get(id);
    if (kids) {
      for (const kid of kids) {
        if (!visible.has(kid)) {
          visible.add(kid);
          queue.push(kid);
        }
      }
    }
  }
  return visible;
}

/**
 * Edges with visible targets. If an edge's source is hidden, remap it to the
 * nearest visible ancestor and mark the edge as contracted.
 */
export function computeVisibleEdges(
  tree: RefinementTree,
  visibleNodeIds: Set<string>,
): VisibleEdge[] {
  const parentsOf = new Map<string, string[]>();
  for (const e of tree.edges) {
    let arr = parentsOf.get(e.targetId);
    if (!arr) { arr = []; parentsOf.set(e.targetId, arr); }
    arr.push(e.sourceId);
  }

  function findVisibleAncestor(nodeId: string): string | null {
    if (visibleNodeIds.has(nodeId)) return nodeId;
    const visited = new Set<string>();
    const stack = [nodeId];
    while (stack.length > 0) {
      const cur = stack.pop()!;
      if (visited.has(cur)) continue;
      visited.add(cur);
      const parents = parentsOf.get(cur);
      if (!parents) continue;
      for (const p of parents) {
        if (visibleNodeIds.has(p)) return p;
        stack.push(p);
      }
    }
    return null;
  }

  const edgeMap = new Map<string, VisibleEdge>();
  for (const e of tree.edges) {
    if (!visibleNodeIds.has(e.targetId)) continue;

    let sourceId = e.sourceId;
    let contracted = false;
    if (!visibleNodeIds.has(sourceId)) {
      const ancestor = findVisibleAncestor(sourceId);
      if (!ancestor || ancestor === e.targetId) continue;
      sourceId = ancestor;
      contracted = true;
    }

    const key = `${sourceId}->${e.targetId}`;
    if (!edgeMap.has(key)) {
      edgeMap.set(key, contracted ? { ...e, sourceId, contracted } : e);
    }
  }
  return Array.from(edgeMap.values());
}

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
