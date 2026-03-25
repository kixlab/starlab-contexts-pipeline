import type { RefinementTree, TreeNode, TreeEdge } from "./treeBuilder";
import type { NodePosition } from "../types";

// ---------------------------------------------------------------------------
// Phase A: Compute a deterministic global order for EVERY node (runs once).
// ---------------------------------------------------------------------------

/**
 * Assigns each node a stable `globalOrder` within its sizeBand layer.
 * Layers are processed top-to-bottom: for each layer, nodes are sorted by
 * the median parent globalOrder (ties broken by quality then id).
 * This order never changes once computed — progressive disclosure only
 * filters which nodes are rendered, never re-sorts them.
 */
export function computeGlobalOrder(
  tree: RefinementTree,
  qualityMetric: "D" | "eta2",
): Map<string, number> {
  if (tree.nodes.length === 0) return new Map();

  // Build adjacency
  const parentsOf = new Map<string, string[]>();
  const childrenOf = new Map<string, string[]>();
  for (const e of tree.edges) {
    let p = parentsOf.get(e.targetId);
    if (!p) { p = []; parentsOf.set(e.targetId, p); }
    p.push(e.sourceId);
    let c = childrenOf.get(e.sourceId);
    if (!c) { c = []; childrenOf.set(e.sourceId, c); }
    c.push(e.targetId);
  }

  // Group nodes by sizeBand, sorted ascending
  const layerMap = new Map<number, TreeNode[]>();
  for (const n of tree.nodes) {
    let arr = layerMap.get(n.sizeBand);
    if (!arr) { arr = []; layerMap.set(n.sizeBand, arr); }
    arr.push(n);
  }
  const sortedBands = Array.from(layerMap.keys()).sort((a, b) => a - b);

  const order = new Map<string, number>();

  for (const band of sortedBands) {
    const layer = layerMap.get(band)!;

    // Compute sort key: median of parent orders (or -1 for roots)
    const barycenters = new Map<string, number>();
    for (const n of layer) {
      const parents = parentsOf.get(n.id) ?? [];
      const parentOrders = parents
        .map((pid) => order.get(pid))
        .filter((o) => o !== undefined) as number[];

      if (parentOrders.length > 0) {
        parentOrders.sort((a, b) => a - b);
        const mid = Math.floor(parentOrders.length / 2);
        const median =
          parentOrders.length % 2 === 1
            ? parentOrders[mid]
            : (parentOrders[mid - 1] + parentOrders[mid]) / 2;
        barycenters.set(n.id, median);
      } else {
        barycenters.set(n.id, -1); // root — will be sorted by quality
      }
    }

    // Sort: barycenter first, then quality (higher is better), then id
    layer.sort((a, b) => {
      const bc = (barycenters.get(a.id) ?? 0) - (barycenters.get(b.id) ?? 0);
      if (Math.abs(bc) > 1e-9) return bc;
      const qa = qualityMetric === "D" ? -a.D : a.eta2;
      const qb = qualityMetric === "D" ? -b.D : b.eta2;
      const qd = qb - qa;
      if (Math.abs(qd) > 1e-9) return qd;
      return a.id.localeCompare(b.id);
    });

    for (let i = 0; i < layer.length; i++) {
      order.set(layer[i].id, i);
    }
  }

  return order;
}

// ---------------------------------------------------------------------------
// Phase B: Position visible nodes using pre-computed order (runs per-render).
// ---------------------------------------------------------------------------

const MARGIN = { top: 48, right: 40, bottom: 40, left: 80 };
const NODE_WIDTH = 140;
const NODE_HEIGHT = 60;
const MIN_ROW_HEIGHT = 80;

export { NODE_WIDTH, NODE_HEIGHT, MARGIN };

export function computePositions(
  visibleNodes: TreeNode[],
  visibleEdges: TreeEdge[],
  globalOrder: Map<string, number>,
  containerWidth: number,
  containerHeight: number,
): Map<string, NodePosition> {
  const result = new Map<string, NodePosition>();
  if (visibleNodes.length === 0) return result;

  // Group by sizeBand
  const layerMap = new Map<number, TreeNode[]>();
  for (const n of visibleNodes) {
    let arr = layerMap.get(n.sizeBand);
    if (!arr) { arr = []; layerMap.set(n.sizeBand, arr); }
    arr.push(n);
  }
  const sortedBands = Array.from(layerMap.keys()).sort((a, b) => a - b);
  const numLayers = sortedBands.length;

  // Sort each layer by globalOrder (stable)
  for (const [, layer] of layerMap) {
    layer.sort((a, b) => (globalOrder.get(a.id) ?? 0) - (globalOrder.get(b.id) ?? 0));
  }

  // Compute widest layer for sizing
  let maxInLayer = 1;
  for (const [, layer] of layerMap) {
    maxInLayer = Math.max(maxInLayer, layer.length);
  }

  const usableWidth = Math.max(
    containerWidth - MARGIN.left - MARGIN.right,
    maxInLayer * (NODE_WIDTH + 40),
  );

  // Y positions
  const rowHeight = Math.max(
    MIN_ROW_HEIGHT,
    numLayers > 0 ? containerHeight / Math.min(5, numLayers) : MIN_ROW_HEIGHT,
  );
  const usableHeight = Math.max(
    containerHeight - MARGIN.top - MARGIN.bottom,
    (numLayers - 1) * rowHeight,
  );

  const layerIndex = new Map<number, number>();
  sortedBands.forEach((band, i) => layerIndex.set(band, i));

  for (const [band, layer] of layerMap) {
    const li = layerIndex.get(band)!;
    const y =
      numLayers === 1
        ? MARGIN.top + usableHeight / 2
        : MARGIN.top + (li / (numLayers - 1)) * usableHeight;

    const count = layer.length;
    for (let i = 0; i < count; i++) {
      const x =
        count === 1
          ? MARGIN.left + usableWidth / 2
          : MARGIN.left + (i / (count - 1)) * usableWidth;
      result.set(layer[i].id, { x, y, layer: li });
    }
  }

  // Insert dummy nodes for long edges (spanning 2+ layers)
  let dummyCount = 0;
  for (const e of visibleEdges) {
    const sp = result.get(e.sourceId);
    const tp = result.get(e.targetId);
    if (!sp || !tp) continue;
    if (tp.layer - sp.layer <= 1) continue;
    // Add dummy waypoints at intermediate layers
    for (let l = sp.layer + 1; l < tp.layer; l++) {
      const frac = (l - sp.layer) / (tp.layer - sp.layer);
      const dx = sp.x + (tp.x - sp.x) * frac;
      const dy = sp.y + (tp.y - sp.y) * frac;
      const dummyId = `__dummy__${dummyCount++}`;
      result.set(dummyId, { x: dx, y: dy, layer: l, isDummy: true });
    }
  }

  return result;
}
