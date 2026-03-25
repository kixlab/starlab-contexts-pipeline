import type { RefinementTree, TreeNode, TreeEdge } from "./treeBuilder";
import type { NodePosition } from "../types";

// Re-export constants and types so page.tsx / TreeCanvas can import from here.
export { NODE_WIDTH, NODE_HEIGHT, MARGIN } from "./stableLayout";
export type { LayoutResult } from "./stableLayout";

// Local copies for use in computePositions
const _NODE_WIDTH = 160;
const _MARGIN = { top: 48, right: 40, bottom: 40, left: 80 };
const _NODE_HEIGHT = 60;
const MIN_ROW_HEIGHT = 80;
const NODE_GAP = 20; // min gap between node edges

type LayoutResult = {
  positions: Map<string, NodePosition>;
  bandRows: { band: number; y: number; count: number }[];
  rowHeight: number;
};

/**
 * Compute a global order for every node, sorted by D ascending within each
 * sizeBand layer. Low D (best effectiveness) is placed leftmost.
 */
export function computeDOrder(tree: RefinementTree): Map<string, number> {
  if (tree.nodes.length === 0) return new Map();

  // Group nodes by sizeBand
  const layerMap = new Map<number, TreeNode[]>();
  for (const n of tree.nodes) {
    let arr = layerMap.get(n.sizeBand);
    if (!arr) { arr = []; layerMap.set(n.sizeBand, arr); }
    arr.push(n);
  }

  const order = new Map<string, number>();

  for (const [, layer] of layerMap) {
    // Sort: D ascending (best = left), then id tiebreak
    layer.sort((a, b) => {
      const dDiff = a.D - b.D;
      if (Math.abs(dDiff) > 1e-9) return dDiff;
      return a.id.localeCompare(b.id);
    });

    for (let i = 0; i < layer.length; i++) {
      order.set(layer[i].id, i);
    }
  }

  return order;
}

/**
 * Position nodes using D for absolute x-positioning across all rows.
 * Nodes with the same D value land at the same x regardless of which row
 * they are in. Collision resolution pushes overlapping nodes apart within
 * each row while preserving order.
 *
 * Y-axis: bands (sizeBand) → rows, same as stableLayout.
 */
export function computePositions(
  visibleNodes: TreeNode[],
  visibleEdges: TreeEdge[],
  globalOrder: Map<string, number>,
  containerWidth: number,
  containerHeight: number,
  allBands?: number[],
): LayoutResult {
  const positions = new Map<string, NodePosition>();
  if (visibleNodes.length === 0 && (!allBands || allBands.length === 0)) {
    return { positions, bandRows: [], rowHeight: MIN_ROW_HEIGHT };
  }

  // ── Group visible nodes by sizeBand ──
  const layerMap = new Map<number, TreeNode[]>();
  for (const n of visibleNodes) {
    let arr = layerMap.get(n.sizeBand);
    if (!arr) { arr = []; layerMap.set(n.sizeBand, arr); }
    arr.push(n);
  }

  const sortedBands = allBands
    ? [...new Set(allBands)].sort((a, b) => a - b)
    : Array.from(layerMap.keys()).sort((a, b) => a - b);
  const numLayers = sortedBands.length;

  // ── Sort each layer by D ascending (via globalOrder) ──
  for (const [, layer] of layerMap) {
    layer.sort((a, b) => (globalOrder.get(a.id) ?? 0) - (globalOrder.get(b.id) ?? 0));
  }

  // ── D range across ALL visible nodes ──
  let minD = Infinity, maxD = -Infinity;
  for (const n of visibleNodes) {
    minD = Math.min(minD, n.D);
    maxD = Math.max(maxD, n.D);
  }
  if (maxD - minD < 1e-9) { minD = 0; maxD = Math.max(1, maxD); }

  // ── Usable width: ensure room for the widest layer after collision ──
  let maxInLayer = 1;
  for (const [, layer] of layerMap) {
    maxInLayer = Math.max(maxInLayer, layer.length);
  }
  const minSpacing = _NODE_WIDTH + NODE_GAP;
  const usableWidth = Math.max(
    containerWidth - _MARGIN.left - _MARGIN.right,
    maxInLayer * minSpacing,
  );

  // ── Y positions ──
  const rowHeight = Math.max(
    MIN_ROW_HEIGHT,
    numLayers > 0 ? containerHeight / Math.min(5, numLayers) : MIN_ROW_HEIGHT,
  );
  const usableHeight = Math.max(
    containerHeight - _MARGIN.top - _MARGIN.bottom,
    (numLayers - 1) * rowHeight,
  );

  const layerIndex = new Map<number, number>();
  sortedBands.forEach((band, i) => layerIndex.set(band, i));

  const bandRows: LayoutResult["bandRows"] = [];
  for (let bi = 0; bi < sortedBands.length; bi++) {
    const band = sortedBands[bi];
    const y =
      numLayers === 1
        ? _MARGIN.top + usableHeight / 2
        : _MARGIN.top + (bi / (numLayers - 1)) * usableHeight;
    const layer = layerMap.get(band);
    bandRows.push({ band, y, count: layer?.length ?? 0 });
  }

  // ── X positions: D-based with collision resolution ──
  for (const [band, layer] of layerMap) {
    const li = layerIndex.get(band)!;
    const y = bandRows[li].y;

    // Ideal x from D (low D → left, high D → right)
    // Offset by NODE_WIDTH/2 so the leftmost node's edge aligns with the margin
    const xStart = _MARGIN.left + _NODE_WIDTH / 2;
    const xRange = usableWidth - _NODE_WIDTH;
    const xs: number[] = layer.map((n) => {
      const frac = (n.D - minD) / (maxD - minD);
      return xStart + frac * Math.max(0, xRange);
    });

    // Collision resolution: sweep left→right, enforce min spacing
    for (let i = 1; i < xs.length; i++) {
      if (xs[i] - xs[i - 1] < minSpacing) {
        xs[i] = xs[i - 1] + minSpacing;
      }
    }

    for (let i = 0; i < layer.length; i++) {
      positions.set(layer[i].id, { x: xs[i], y, layer: li });
    }
  }

  // ── Dummy nodes for long edges ──
  let dummyCount = 0;
  for (const e of visibleEdges) {
    const sp = positions.get(e.sourceId);
    const tp = positions.get(e.targetId);
    if (!sp || !tp) continue;
    if (tp.layer - sp.layer <= 1) continue;
    for (let l = sp.layer + 1; l < tp.layer; l++) {
      const frac = (l - sp.layer) / (tp.layer - sp.layer);
      const dx = sp.x + (tp.x - sp.x) * frac;
      const dy = sp.y + (tp.y - sp.y) * frac;
      const dummyId = `__dummy__${dummyCount++}`;
      positions.set(dummyId, { x: dx, y: dy, layer: l, isDummy: true });
    }
  }

  return { positions, bandRows, rowHeight };
}
