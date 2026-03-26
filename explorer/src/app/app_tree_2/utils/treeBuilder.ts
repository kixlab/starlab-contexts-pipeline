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

export type EdgeType = "add" | "restructure";

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

/** Encode string keys → Uint32Array ids + compute entropy in one pass. */
function encodeAndEntropy(keys: string[]): {
  ids: Uint32Array;
  cardinality: number;
  H: number;
} {
  const N = keys.length;
  if (N === 0) return { ids: new Uint32Array(0), cardinality: 0, H: 0 };

  const dict = new Map<string, number>();
  const ids = new Uint32Array(N);
  const counts: number[] = [];
  let nextId = 0;

  for (let i = 0; i < N; i++) {
    let id = dict.get(keys[i]);
    if (id === undefined) {
      id = nextId++;
      dict.set(keys[i], id);
      counts.push(0);
    }
    ids[i] = id;
    counts[id]++;
  }

  let H = 0;
  for (let i = 0; i < counts.length; i++) {
    const p = counts[i] / N;
    if (p > 0) H -= p * Math.log2(p);
  }

  return { ids, cardinality: nextId, H };
}

// ---------------------------------------------------------------------------
// MI computation (exported for InspectionPanel)
// ---------------------------------------------------------------------------

/**
 * Mutual information MI(A, B) = H(A) + H(B) - H(A,B).
 * Uses typed array for joint counting when cardA*cardB < 50000.
 */
export function computeMI(a: TreeNode, b: TreeNode, N: number): number {
  if (N === 0) return 0;
  const product = a.partitionCardinality * b.partitionCardinality;
  let H_joint: number;

  if (product < 50_000) {
    // Fast path: typed array
    const joint = new Uint32Array(product);
    const cardB = b.partitionCardinality;
    for (let i = 0; i < N; i++) {
      joint[a.partitionIds[i] * cardB + b.partitionIds[i]]++;
    }
    H_joint = 0;
    for (let i = 0; i < product; i++) {
      if (joint[i] > 0) {
        const p = joint[i] / N;
        H_joint -= p * Math.log2(p);
      }
    }
  } else {
    // Fallback: Map
    const joint = new Map<number, number>();
    const cardB = b.partitionCardinality;
    for (let i = 0; i < N; i++) {
      const k = a.partitionIds[i] * cardB + b.partitionIds[i];
      joint.set(k, (joint.get(k) ?? 0) + 1);
    }
    H_joint = 0;
    for (const c of joint.values()) {
      const p = c / N;
      if (p > 0) H_joint -= p * Math.log2(p);
    }
  }

  return Math.max(0, a.H + b.H - H_joint);
}

/** Joint entropy H(A,B). Used by InspectionPanel comparison mode. */
export function computeJointEntropy(a: TreeNode, b: TreeNode, N: number): number {
  if (N === 0) return 0;
  const product = a.partitionCardinality * b.partitionCardinality;

  if (product < 50_000) {
    const joint = new Uint32Array(product);
    const cardB = b.partitionCardinality;
    for (let i = 0; i < N; i++) {
      joint[a.partitionIds[i] * cardB + b.partitionIds[i]]++;
    }
    let H = 0;
    for (let i = 0; i < product; i++) {
      if (joint[i] > 0) {
        const p = joint[i] / N;
        H -= p * Math.log2(p);
      }
    }
    return H;
  }

  const joint = new Map<number, number>();
  const cardB = b.partitionCardinality;
  for (let i = 0; i < N; i++) {
    const k = a.partitionIds[i] * cardB + b.partitionIds[i];
    joint.set(k, (joint.get(k) ?? 0) + 1);
  }
  let H = 0;
  for (const c of joint.values()) {
    const p = c / N;
    if (p > 0) H -= p * Math.log2(p);
  }
  return H;
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

    // --- Compute D (H(U|C)) inline ---
    let totalItems = 0;
    for (const uc of cellUnits.values()) {
      for (const c of uc.values()) totalItems += c;
    }
    let D = 0;
    if (totalItems > 0) {
      for (const uc of cellUnits.values()) {
        let mc = 0;
        for (const c of uc.values()) mc += c;
        if (mc === 0) continue;
        const pc = mc / totalItems;
        let hc = 0;
        for (const c of uc.values()) {
          if (c > 0) {
            const pu = c / mc;
            hc -= pu * Math.log2(pu);
          }
        }
        D += pc * hc;
      }
    }

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

    // --- Encode partition + entropy in one pass ---
    const { ids: partitionIds, cardinality: partitionCardinality, H } =
      encodeAndEntropy(partitionKeys);

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
  return { allNodes, validPiecesCount: rows.length, numUniqueUnits: unitIds.size };
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
  const N = cached.validPiecesCount;

  // Step 1: filter by stability
  const nodes = cached.allNodes
    .filter((n) => n.stability >= config.stabilityThreshold)
    .map((n) => ({ ...n, sizeBand: n.size }));

  // Step 2+3: prune & connect
  const { survivors, edges: rawEdges } = pruneAndConnect(nodes, config, N);

  // Step 4: transitive reduction
  const reduced = transitiveReduction(rawEdges);

  // Step 5: annotate + find roots
  const nodeMap = new Map(survivors.map((n) => [n.id, n]));
  const edges = annotateEdges(reduced, nodeMap);
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
  N: number,
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
      v = computeMI(a, b, N);
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
  // Build adjacency
  const adj = new Map<string, Set<string>>();
  for (const e of edges) {
    let s = adj.get(e.sourceId);
    if (!s) { s = new Set(); adj.set(e.sourceId, s); }
    s.add(e.targetId);
  }

  // For each edge A→B, check if B is reachable from A via other paths
  return edges.filter((e) => {
    const children = adj.get(e.sourceId);
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
      const next = adj.get(cur);
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
        : "restructure";
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
