import type { Facet, DatasetVideo } from "@/app/types";
import { facetToLabelKey, normalizeFacetVals } from "@/app/utils/facetStats";
import {
  computeCompactness,
  CONTENT_TYPE_FACET_ID,
  MAX_FACETS_CAP,
} from "@/app/app_analysis/utils/metricsUtils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TreeNode = {
  id: string;
  facetSubset: Facet[];
  /** Compactness C — total non-NA vocab items across facets. */
  size: number;
  /** Layer key (= size). */
  sizeBand: number;
  /** Discriminativeness H(U|C). Lower is better. */
  D: number;
  /** Temporal variance explained (eta-squared). Higher is better. */
  eta2: number;
  /** Transition stability. Higher is better. */
  stability: number;
  cellCount: number;
  /** Compact numeric partition (one entry per valid piece). */
  partitionIds: Uint32Array;
  partitionCardinality: number;
  /** Partition entropy H. */
  H: number;
};

export type EdgeType = "add" | "refine";

export type TreeEdge = {
  sourceId: string;
  targetId: string;
  MI: number;
  /** MI / H(source) — asymmetric capture ratio. */
  refinementScore: number;
  marginalDGain: number;
  marginalEta2Gain: number;
  addedFacets: Facet[];
  removedFacets: Facet[];
  edgeType: EdgeType;
};

export type RefinementTree = {
  nodes: TreeNode[];
  edges: TreeEdge[];
  roots: string[];
};

export type TreeBuildConfig = {
  maxK: number;
  qualityMetric: "D" | "eta2";
  stabilityThreshold: number;
  /** τ — MI/H threshold for capture / edge creation. */
  refinementThreshold: number;
};

export const DEFAULT_TREE_CONFIG: TreeBuildConfig = {
  maxK: 2,
  qualityMetric: "D",
  stabilityThreshold: 0,
  refinementThreshold: 0.6,
};

export type CachedSchemas = {
  allNodes: TreeNode[];
  validPiecesCount: number;
  numUniqueUnits: number;
  /** Per-piece encoded unit ID (index into unitTotals). */
  pieceUnitIds: Uint32Array;
  /** Total piece count per unit (indexed by encoded unit ID). */
  unitTotals: Uint32Array;
};

/** Organization Effectiveness: percentage (0–100), higher = better discrimination. */
export function computeEffectiveness(D: number, numUniqueUnits: number): number {
  if (numUniqueUnits <= 1) return 0;
  const maxH = Math.log2(numUniqueUnits);
  return maxH > 0 ? Math.max(0, Math.min(100, (1 - D / maxH) * 100)) : 0;
}

// ---------------------------------------------------------------------------
// Low-level helpers
// ---------------------------------------------------------------------------

function variance(vals: number[]): number {
  const n = vals.length;
  if (n <= 1) return 0;
  const mean = vals.reduce((a, b) => a + b, 0) / n;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += (vals[i] - mean) ** 2;
  return sum / n;
}

/** Generate all C(n,k) combinations as arrays of indices. */
function combinations(n: number, k: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  (function go(start: number) {
    if (cur.length === k) { out.push(cur.slice()); return; }
    const rem = k - cur.length;
    for (let i = start; i <= n - rem; i++) {
      cur.push(i);
      go(i + 1);
      cur.pop();
    }
  })(0);
  return out;
}

/** Encode string keys → Uint32Array ids (no entropy — use unit-weighted helpers). */
function encodePartition(keys: string[]): {
  ids: Uint32Array;
  cardinality: number;
} {
  const N = keys.length;
  if (N === 0) return { ids: new Uint32Array(0), cardinality: 0 };

  const dict = new Map<string, number>();
  const ids = new Uint32Array(N);
  let nextId = 0;

  for (let i = 0; i < N; i++) {
    let id = dict.get(keys[i]);
    if (id === undefined) {
      id = nextId++;
      dict.set(keys[i], id);
    }
    ids[i] = id;
  }

  return { ids, cardinality: nextId };
}

/**
 * Unit-weighted entropy H(S).
 *
 *   P(S=s) = (1/numUnits) * Σ_u (count_{u,s} / total_u)
 *   H      = -Σ_s P(s) * log₂(P(s))
 */
function computeUnitWeightedEntropy(
  partitionIds: Uint32Array,
  partitionCardinality: number,
  pieceUnitIds: Uint32Array,
  unitTotals: Uint32Array,
  numUnits: number,
): number {
  if (numUnits === 0 || partitionCardinality === 0) return 0;
  const N = partitionIds.length;

  // For each partition cell, accumulate per-unit counts:
  // cellUnitCounts[cell * numUnits + unit] = count
  const cellUnitCounts = new Float64Array(partitionCardinality * numUnits);
  for (let i = 0; i < N; i++) {
    cellUnitCounts[partitionIds[i] * numUnits + pieceUnitIds[i]]++;
  }

  let H = 0;
  for (let s = 0; s < partitionCardinality; s++) {
    let wSum = 0;
    const base = s * numUnits;
    for (let u = 0; u < numUnits; u++) {
      const c = cellUnitCounts[base + u];
      if (c > 0) wSum += c / unitTotals[u];
    }
    const pS = wSum / numUnits;
    if (pS > 0) H -= pS * Math.log2(pS);
  }
  return H;
}

/**
 * Unit-weighted discriminativeness D = H(U|C).
 *
 *   w_u(c)      = m_{u,c} / m_{u,*}
 *   P(C=c)      = (1/N) * Σ_u w_u(c)
 *   P(U=u|C=c)  = w_u(c) / Σ_{u'} w_{u'}(c)
 *   D            = Σ_c P(C=c) * H(U|C=c)
 */
function computeUnitWeightedD(
  cellUnits: Map<string, Map<string, number>>,
  globalUnitTotals: Map<string, number>,
): number {
  const numUnits = globalUnitTotals.size;
  if (numUnits === 0) return 0;

  let D = 0;
  for (const unitMap of cellUnits.values()) {
    // Compute weights w_u = m_{u,c} / m_{u,*}
    const weights: number[] = [];
    const entries: [string, number][] = [];
    for (const [uid, count] of unitMap) {
      if (count > 0) {
        const w = count / globalUnitTotals.get(uid)!;
        weights.push(w);
        entries.push([uid, w]);
      }
    }
    const wSum = weights.reduce((a, b) => a + b, 0);
    if (wSum === 0) continue;

    const pC = wSum / numUnits;
    let hC = 0;
    for (const w of weights) {
      const pUgivenC = w / wSum;
      if (pUgivenC > 0) hC -= pUgivenC * Math.log2(pUgivenC);
    }
    D += pC * hC;
  }
  return D;
}

// ---------------------------------------------------------------------------
// MI computation (exported for InspectionPanel)
// ---------------------------------------------------------------------------

/**
 * Unit-weighted joint entropy H(A,B).
 *
 *   P(A=a,B=b) = (1/numUnits) * Σ_u (count_{u,(a,b)} / total_u)
 *
 * Shared implementation for computeMI and computeJointEntropy.
 */
function unitWeightedJointEntropy(
  a: TreeNode,
  b: TreeNode,
  pieceUnitIds: Uint32Array,
  unitTotals: Uint32Array,
  numUnits: number,
): number {
  const N = pieceUnitIds.length;
  if (N === 0 || numUnits === 0) return 0;
  const cardB = b.partitionCardinality;
  const product = a.partitionCardinality * cardB;

  // Accumulate per-unit counts for each joint cell.
  // jointUnitCounts[(a*cardB + b) * numUnits + u] = count
  if (product * numUnits < 500_000) {
    // Fast path: flat typed array
    const jointUC = new Float64Array(product * numUnits);
    for (let i = 0; i < N; i++) {
      const jCell = a.partitionIds[i] * cardB + b.partitionIds[i];
      jointUC[jCell * numUnits + pieceUnitIds[i]]++;
    }
    let H = 0;
    for (let j = 0; j < product; j++) {
      let wSum = 0;
      const base = j * numUnits;
      for (let u = 0; u < numUnits; u++) {
        const c = jointUC[base + u];
        if (c > 0) wSum += c / unitTotals[u];
      }
      const p = wSum / numUnits;
      if (p > 0) H -= p * Math.log2(p);
    }
    return H;
  }

  // Fallback: Map of joint cell → Map<unit, count>
  const jointMap = new Map<number, Float64Array>();
  for (let i = 0; i < N; i++) {
    const jCell = a.partitionIds[i] * cardB + b.partitionIds[i];
    let uc = jointMap.get(jCell);
    if (!uc) { uc = new Float64Array(numUnits); jointMap.set(jCell, uc); }
    uc[pieceUnitIds[i]]++;
  }
  let H = 0;
  for (const uc of jointMap.values()) {
    let wSum = 0;
    for (let u = 0; u < numUnits; u++) {
      if (uc[u] > 0) wSum += uc[u] / unitTotals[u];
    }
    const p = wSum / numUnits;
    if (p > 0) H -= p * Math.log2(p);
  }
  return H;
}

/**
 * Unit-weighted mutual information MI(A, B) = H(A) + H(B) - H(A,B).
 */
export function computeMI(
  a: TreeNode,
  b: TreeNode,
  pieceUnitIds: Uint32Array,
  unitTotals: Uint32Array,
  numUnits: number,
): number {
  const H_joint = unitWeightedJointEntropy(a, b, pieceUnitIds, unitTotals, numUnits);
  return Math.max(0, a.H + b.H - H_joint);
}

/** Unit-weighted joint entropy H(A,B). Used by InspectionPanel comparison mode. */
export function computeJointEntropy(
  a: TreeNode,
  b: TreeNode,
  pieceUnitIds: Uint32Array,
  unitTotals: Uint32Array,
  numUnits: number,
): number {
  return unitWeightedJointEntropy(a, b, pieceUnitIds, unitTotals, numUnits);
}

// ---------------------------------------------------------------------------
// Phase 1: Enumerate all facet-subset schemas
// ---------------------------------------------------------------------------

type PieceRow = {
  components: string[]; // one string per facet (pre-computed)
  contentType: string;
  unitId: string;
  videoId: string;
  position: number;
};

export async function enumerateSchemas(
  dataset: DatasetVideo[],
  allFacets: Facet[],
  infoTypes: string[] | null,
  maxK: number,
  onProgress?: (step: string, done: number, total: number) => void,
): Promise<CachedSchemas | null> {
  // Sort facets deterministically
  const facets = [...allFacets].sort((a, b) =>
    facetToLabelKey(a).localeCompare(facetToLabelKey(b)),
  );
  const facetKeys = facets.map(facetToLabelKey);

  // Pre-compute piece rows (one-time O(P×F) pass)
  const rows: PieceRow[] = [];
  const unitIds = new Set<string>();

  for (const video of dataset) {
    const videoId = video.id ?? video.url;
    const pieces = video.pieces ?? [];
    const filtered =
      infoTypes && infoTypes.length > 0
        ? pieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
        : pieces;
    const sorted = [...filtered].sort((a, b) => a.start - b.start);
    let pos = 0;
    for (const p of sorted) {
      if (!p.unit_id) continue;
      unitIds.add(p.unit_id as string);
      const labels = (p.labels ?? {}) as Record<string, string[]>;
      const components = facetKeys.map((k) => {
        if (k === CONTENT_TYPE_FACET_ID) return p.content_type ?? "unknown";
        return normalizeFacetVals(labels[k] as string[] | undefined).sort().join("|");
      });
      rows.push({
        components,
        contentType: p.content_type ?? "unknown",
        unitId: p.unit_id as string,
        videoId,
        position: pos,
      });
      pos++;
    }
  }

  if (rows.length === 0 || unitIds.size === 0) return null;

  // Encode unit IDs → Uint32Array + compute per-unit totals
  const unitDict = new Map<string, number>();
  let nextUnitId = 0;
  const pieceUnitIds = new Uint32Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    let uid = unitDict.get(rows[i].unitId);
    if (uid === undefined) {
      uid = nextUnitId++;
      unitDict.set(rows[i].unitId, uid);
    }
    pieceUnitIds[i] = uid;
  }
  const numUnits = nextUnitId;
  const unitTotals = new Uint32Array(numUnits);
  for (let i = 0; i < rows.length; i++) unitTotals[pieceUnitIds[i]]++;

  // Also build a string-keyed unit totals map for D computation
  const globalUnitTotals = new Map<string, number>();
  for (const [uid, idx] of unitDict) globalUnitTotals.set(uid, unitTotals[idx]);

  // Generate all combos 1..maxK
  const allCombos: number[][] = [];
  for (let k = 1; k <= Math.min(maxK, facets.length); k++) {
    allCombos.push(...combinations(facets.length, k));
  }

  const allNodes: TreeNode[] = [];
  const YIELD_EVERY = 50;

  for (let ci = 0; ci < allCombos.length; ci++) {
    const fi = allCombos[ci]; // facet indices for this combo
    const subset = fi.map((i) => facets[i]);
    const id = fi.map((i) => facetKeys[i]).sort().join("+");
    const size = computeCompactness(subset);

    // --- Single pass over rows: build partition keys, cells, per-video groups ---
    const hasContentType = fi.some((i) => facetKeys[i] === CONTENT_TYPE_FACET_ID);

    const partitionKeys: string[] = new Array(rows.length);
    // Cell → unit → count (for D computation)
    const cellUnits = new Map<string, Map<string, number>>();
    // Video → array of { position, key } (for eta2 + stability)
    const videoGroups = new Map<string, { position: number; key: string }[]>();

    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      const parts = fi.map((i) => `${facetKeys[i]}:${row.components[i]}`).sort();
      const key = hasContentType ? parts.join(";") : `${row.contentType}|${parts.join(";")}`;
      partitionKeys[r] = key;

      // Accumulate cell→unit counts
      let unitMap = cellUnits.get(key);
      if (!unitMap) { unitMap = new Map(); cellUnits.set(key, unitMap); }
      unitMap.set(row.unitId, (unitMap.get(row.unitId) ?? 0) + 1);

      // Accumulate video groups
      let vg = videoGroups.get(row.videoId);
      if (!vg) { vg = []; videoGroups.set(row.videoId, vg); }
      vg.push({ position: row.position, key });
    }

    // --- Compute D (H(U|C)) using unit-weighted probabilities ---
    const D = computeUnitWeightedD(cellUnits, globalUnitTotals);

    // --- Compute eta2 and stability from videoGroups ---
    let eta2Sum = 0, eta2Count = 0;
    let stabSum = 0, stabCount = 0;

    for (const items of videoGroups.values()) {
      const n = items.length;
      if (n <= 1) continue;

      // Stability: count transitions
      let transitions = 0;
      for (let i = 0; i < n - 1; i++) {
        if (items[i].key !== items[i + 1].key) transitions++;
      }
      stabSum += transitions / (n - 1);
      stabCount++;

      // Eta2: 1 - withinVar/totalVar
      const allPos = items.map((it) => it.position);
      const totalVar = variance(allPos);
      if (totalVar === 0) continue;

      const groups = new Map<string, number[]>();
      for (const it of items) {
        let g = groups.get(it.key);
        if (!g) { g = []; groups.set(it.key, g); }
        g.push(it.position);
      }
      let withinVar = 0;
      for (const positions of groups.values()) {
        withinVar += (positions.length / n) * variance(positions);
      }
      eta2Sum += 1 - withinVar / totalVar;
      eta2Count++;
    }

    const eta2 = eta2Count > 0 ? eta2Sum / eta2Count : 0;
    const stability = 1 - (stabCount > 0 ? stabSum / stabCount : 0);

    // --- Encode partition + compute unit-weighted entropy ---
    const { ids: partitionIds, cardinality: partitionCardinality } =
      encodePartition(partitionKeys);
    const H = computeUnitWeightedEntropy(
      partitionIds, partitionCardinality, pieceUnitIds, unitTotals, numUnits,
    );

    allNodes.push({
      id,
      facetSubset: subset,
      size,
      sizeBand: size,
      D,
      eta2,
      stability,
      cellCount: cellUnits.size,
      partitionIds,
      partitionCardinality,
      H,
    });

    // Yield to event loop periodically
    if ((ci + 1) % YIELD_EVERY === 0) {
      onProgress?.("Enumerating schemas", ci + 1, allCombos.length);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }

  onProgress?.("Enumerating schemas", allCombos.length, allCombos.length);
  return {
    allNodes,
    validPiecesCount: rows.length,
    numUniqueUnits: unitIds.size,
    pieceUnitIds,
    unitTotals,
  };
}

// ---------------------------------------------------------------------------
// Phase 2: Build tree from cached schemas
// ---------------------------------------------------------------------------

/** Quality value (higher = better for both metrics). */
function nodeQ(n: TreeNode, metric: "D" | "eta2"): number {
  return metric === "D" ? -n.D : n.eta2;
}

export function buildTreeFromCachedSchemas(
  cached: CachedSchemas,
  config: TreeBuildConfig,
): RefinementTree {
  const { pieceUnitIds, unitTotals, numUniqueUnits: numUnits } = cached;

  // Step 1: filter by stability
  const nodes = cached.allNodes
    .filter((n) => n.stability >= config.stabilityThreshold)
    .map((n) => ({ ...n, sizeBand: n.size }));

  // Step 2+3: prune & connect
  const { survivors, edges: rawEdges } = pruneAndConnect(
    nodes, config, pieceUnitIds, unitTotals, numUnits,
  );

  // Step 4: annotate edges (before reduction so edge types are available)
  const nodeMap = new Map(survivors.map((n) => [n.id, n]));
  const annotated = annotateEdges(rawEdges, nodeMap);

  // Step 5: type-aware transitive reduction
  const edges = transitiveReduction(annotated);
  const incoming = new Set(edges.map((e) => e.targetId));
  const roots = survivors.filter((n) => !incoming.has(n.id)).map((n) => n.id);

  return { nodes: survivors, edges, roots };
}

// ---------------------------------------------------------------------------
// Prune & connect (greedy survival + edge building with MI cache)
// ---------------------------------------------------------------------------

function pruneAndConnect(
  nodes: TreeNode[],
  config: TreeBuildConfig,
  pieceUnitIds: Uint32Array,
  unitTotals: Uint32Array,
  numUnits: number,
): { survivors: TreeNode[]; edges: TreeEdge[] } {
  const τ = config.refinementThreshold;

  // Sort: smaller size first → better quality first → id tiebreak
  const sorted = [...nodes].sort((a, b) => {
    if (a.size !== b.size) return a.size - b.size;
    const qd = nodeQ(b, config.qualityMetric) - nodeQ(a, config.qualityMetric);
    if (Math.abs(qd) > 1e-9) return qd;
    return a.id.localeCompare(b.id);
  });

  // MI cache (symmetric: key = sorted pair)
  const miCache = new Map<string, number>();
  function cachedMI(a: TreeNode, b: TreeNode): number {
    const key = a.id < b.id ? `${a.id}\0${b.id}` : `${b.id}\0${a.id}`;
    let v = miCache.get(key);
    if (v === undefined) {
      v = computeMI(a, b, pieceUnitIds, unitTotals, numUnits);
      miCache.set(key, v);
    }
    return v;
  }

  // Phase A: greedy survival
  const accepted: TreeNode[] = [];
  for (const candidate of sorted) {
    // Early exit: H=0 means trivially dominated by any keeper of ≤ size
    if (candidate.H === 0 && accepted.some((k) => k.size <= candidate.size)) {
      continue;
    }
    let dominated = false;
    for (const keeper of accepted) {
      if (keeper.size > candidate.size) continue; // can't dominate
      const mi = cachedMI(keeper, candidate);
      const refFromCandidate = candidate.H > 0 ? mi / candidate.H : 0;
      if (refFromCandidate > τ) {
        dominated = true;
        break;
      }
    }
    if (!dominated) accepted.push(candidate);
  }

  // Phase B: build edges among survivors
  const edges: TreeEdge[] = [];
  for (let i = 0; i < accepted.length; i++) {
    for (let j = i + 1; j < accepted.length; j++) {
      const A = accepted[i];
      const B = accepted[j];
      if (A.size === B.size) continue; // edges only between different sizes
      const mi = cachedMI(A, B);
      const refAB = A.H > 0 ? mi / A.H : 0;
      const refBA = B.H > 0 ? mi / B.H : 0;

      if (refAB > τ && A.size < B.size) {
        edges.push(makeEdge(A.id, B.id, mi, refAB));
      }
      if (refBA > τ && B.size < A.size) {
        edges.push(makeEdge(B.id, A.id, mi, refBA));
      }
    }
  }

  return { survivors: accepted, edges };
}

function makeEdge(
  sourceId: string,
  targetId: string,
  MI: number,
  refinementScore: number,
): TreeEdge {
  return {
    sourceId,
    targetId,
    MI,
    refinementScore,
    marginalDGain: 0,
    marginalEta2Gain: 0,
    addedFacets: [],
    removedFacets: [],
    edgeType: "add",
  };
}

// ---------------------------------------------------------------------------
// Transitive reduction
// ---------------------------------------------------------------------------

function transitiveReduction(edges: TreeEdge[]): TreeEdge[] {
  // Build per-type adjacency: only follow edges of the same type during BFS.
  // This preserves edges whose target is reachable only via mixed-type paths,
  // since those paths carry different semantic information.
  const adjByType = new Map<EdgeType, Map<string, Set<string>>>();
  for (const e of edges) {
    let typeAdj = adjByType.get(e.edgeType);
    if (!typeAdj) { typeAdj = new Map(); adjByType.set(e.edgeType, typeAdj); }
    let s = typeAdj.get(e.sourceId);
    if (!s) { s = new Set(); typeAdj.set(e.sourceId, s); }
    s.add(e.targetId);
  }

  // For each edge A→B of type T, check if B is reachable from A
  // via other same-type paths only
  return edges.filter((e) => {
    const typeAdj = adjByType.get(e.edgeType);
    if (!typeAdj) return true;
    const children = typeAdj.get(e.sourceId);
    if (!children || children.size <= 1) return true;
    const visited = new Set<string>();
    const queue: string[] = [];
    for (const child of children) {
      if (child !== e.targetId) queue.push(child);
    }
    while (queue.length > 0) {
      const cur = queue.shift()!;
      if (cur === e.targetId) return false;
      if (visited.has(cur)) continue;
      visited.add(cur);
      const next = typeAdj.get(cur);
      if (next) {
        for (const n of next) {
          if (!visited.has(n)) queue.push(n);
        }
      }
    }
    return true;
  });
}

// ---------------------------------------------------------------------------
// Edge annotation
// ---------------------------------------------------------------------------

function annotateEdges(
  edges: TreeEdge[],
  nodeMap: Map<string, TreeNode>,
): TreeEdge[] {
  return edges.map((e) => {
    const src = nodeMap.get(e.sourceId)!;
    const tgt = nodeMap.get(e.targetId)!;
    const srcIds = new Set(src.facetSubset.map((f) => f.id));
    const tgtIds = new Set(tgt.facetSubset.map((f) => f.id));
    const addedFacets = tgt.facetSubset.filter((f) => !srcIds.has(f.id));
    const removedFacets = src.facetSubset.filter((f) => !tgtIds.has(f.id));
    const edgeType: EdgeType =
      removedFacets.length === 0 && addedFacets.length > 0
        ? "add"
        : "refine";
    return {
      ...e,
      marginalDGain: src.D - tgt.D,
      marginalEta2Gain: tgt.eta2 - src.eta2,
      addedFacets,
      removedFacets,
      edgeType,
    };
  });
}
