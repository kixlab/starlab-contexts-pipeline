import type { RefinementTree, TreeNode, TreeEdge } from "./treeBuilder";
import type { NodePosition } from "../types";

// Re-export constants so page.tsx / TreeCanvas can import from here.
export { NODE_WIDTH, NODE_HEIGHT, MARGIN } from "./stableLayout";

// Local copies for use in computePositions
const _NODE_WIDTH = 160;
const _MARGIN = { top: 48, right: 40, bottom: 40, left: 80 };
const MIN_ROW_HEIGHT = 80;
const NODE_GAP = 20; // min horizontal gap between node edges

// Adaptive node height (must stay in sync with TreeCanvas BASE_H / CHIP_H)
const NODE_H_BASE = 20;
const NODE_H_PER_CHIP = 20;
const SUB_ROW_GAP = 14; // vertical gap between sub-lanes

function computeNodeH(n: TreeNode): number {
  return NODE_H_BASE + n.facetSubset.length * NODE_H_PER_CHIP;
}

export type LayoutResult = {
  positions: Map<string, NodePosition>;
  /** Per-band row info. `height` may differ across bands (sub-lanes). */
  bandRows: { band: number; y: number; count: number; height: number }[];
  rowHeight: number; // max band height (for backwards compat)
};

/**
 * Compute a global order for every node, sorted by D ascending within each
 * sizeBand layer. Low D (best effectiveness) is placed leftmost.
 */
export function computeDOrder(tree: RefinementTree): Map<string, number> {
  if (tree.nodes.length === 0) return new Map();

  const layerMap = new Map<number, TreeNode[]>();
  for (const n of tree.nodes) {
    let arr = layerMap.get(n.sizeBand);
    if (!arr) { arr = []; layerMap.set(n.sizeBand, arr); }
    arr.push(n);
  }

  const order = new Map<string, number>();

  for (const [, layer] of layerMap) {
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
 * When nodes in the same band would overlap at their D-based x, they are
 * placed into sub-lanes (extra vertical rows within the band) instead of
 * widening the canvas. Row heights vary per band.
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
  const dRange = maxD - minD;

  // ── Usable width: moderate, sub-lanes handle density ──
  const usableWidth = Math.max(
    containerWidth - _MARGIN.left - _MARGIN.right,
    800,
  );
  const xStart = _MARGIN.left + _NODE_WIDTH / 2;
  const xRange = usableWidth - _NODE_WIDTH;
  const minSpacing = _NODE_WIDTH + NODE_GAP;

  // ── Assign sub-lanes per band ──
  const nodeLane = new Map<string, number>();
  const bandLaneCount = new Map<number, number>();
  const bandMaxNodeH = new Map<number, number>();

  // Also store each node's ideal x for positioning later
  const nodeIdealX = new Map<string, number>();

  for (const [band, layer] of layerMap) {
    // Each lane tracks the rightmost edge x of its last node
    const lanes: number[] = [];
    let maxH = 0;

    for (const n of layer) {
      const frac = (n.D - minD) / dRange;
      const x = xStart + frac * Math.max(0, xRange);
      nodeIdealX.set(n.id, x);
      const leftEdge = x - _NODE_WIDTH / 2;
      const rightEdge = x + _NODE_WIDTH / 2;
      maxH = Math.max(maxH, computeNodeH(n));

      // Find first lane where this node fits without overlap
      let assigned = false;
      for (let li = 0; li < lanes.length; li++) {
        if (leftEdge >= lanes[li] + NODE_GAP) {
          lanes[li] = rightEdge;
          nodeLane.set(n.id, li);
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        nodeLane.set(n.id, lanes.length);
        lanes.push(rightEdge);
      }
    }

    bandLaneCount.set(band, Math.max(1, lanes.length));
    bandMaxNodeH.set(band, Math.max(maxH, NODE_H_BASE));
  }

  // ── Compute variable row heights and y positions ──
  const bandRows: LayoutResult["bandRows"] = [];
  let currentY = _MARGIN.top;

  for (const band of sortedBands) {
    const numLanes = bandLaneCount.get(band) ?? 1;
    const maxH = bandMaxNodeH.get(band) ?? NODE_H_BASE;
    const subRowH = maxH + SUB_ROW_GAP;
    const bandHeight = Math.max(MIN_ROW_HEIGHT, numLanes * subRowH);
    const bandCenterY = currentY + bandHeight / 2;
    const layer = layerMap.get(band);
    bandRows.push({ band, y: bandCenterY, count: layer?.length ?? 0, height: bandHeight });
    currentY += bandHeight;
  }

  const bandRowIndex = new Map<number, number>();
  sortedBands.forEach((band, i) => bandRowIndex.set(band, i));

  // ── Position each node ──
  for (const [band, layer] of layerMap) {
    const bi = bandRowIndex.get(band)!;
    const bandRow = bandRows[bi];
    const numLanes = bandLaneCount.get(band) ?? 1;
    const maxH = bandMaxNodeH.get(band) ?? NODE_H_BASE;
    const subRowH = maxH + SUB_ROW_GAP;

    for (const n of layer) {
      const x = nodeIdealX.get(n.id)!;
      const lane = nodeLane.get(n.id) ?? 0;
      const laneOffset = (lane - (numLanes - 1) / 2) * subRowH;
      const y = bandRow.y + laneOffset;
      positions.set(n.id, { x, y, layer: bi });
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

  const maxRowHeight = Math.max(...bandRows.map((b) => b.height), MIN_ROW_HEIGHT);
  return { positions, bandRows, rowHeight: maxRowHeight };
}
