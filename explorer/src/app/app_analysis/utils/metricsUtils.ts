import type { Facet, DatasetVideo } from "@/app/types";
import { facetToLabelKey, normalizeFacetVals } from "@/app/utils/facetStats";

export type PlotPoint = {
  datasetLabel: string;
  facetSubset: Facet[];
  /** Sum of vocabulary sizes (excl. "na") — lower = more compact */
  compactness: number;
  /** D = Σ_c (n_c/N)·log₂(n_c) — lower = more discriminative (0 = perfect separation) */
  discriminativeness: number;
  /** Number of distinct cells (cell = content_type + facet-label signature) */
  relativeTotal: number;
  numUniqueUnits: number;
};

export type FacetValueDist = {
  facet: Facet;
  valueCounts: { label: string; count: number; percentage: number }[];
  totalPieces: number;
};

export type VideoBreakdown = {
  videoId: string;
  title: string;
  totalPieces: number;
  uniqueContexts: number;
  contextCounts: { context: string; readableContext: string; count: number }[];
};

// Fixed palette of visually distinct dataset colors
export const DATASET_COLORS = [
  "#E63946",
  "#457B9D",
  "#2A9D8F",
  "#E9A800",
  "#F4A261",
  "#9B5DE5",
  "#00A8C8",
  "#06D6A0",
  "#FB5607",
  "#8338EC",
  "#3A86FF",
  "#FF006E",
  "#06A77D",
  "#D62246",
];

export function getDatasetColor(index: number): string {
  return DATASET_COLORS[index % DATASET_COLORS.length];
}

// ---------------------------------------------------------------------------
// Cell key: content_type + sorted facet-label assignments
// Matches the Python compute_cells() which includes the item subtype in the
// signature alongside facet annotations.
// ---------------------------------------------------------------------------
function buildCellKey(
  piece: DatasetVideo["pieces"][number],
  facetSubset: Facet[]
): string {
  const contentType = piece.content_type ?? "unknown";
  if (facetSubset.length === 0) return contentType;
  const labels = (piece.labels ?? {}) as Record<string, string[]>;
  const parts = facetSubset
    .map((f) => {
      const k = facetToLabelKey(f);
      const raw = (labels[k] ?? []) as string[];
      const vals = normalizeFacetVals(raw);
      return `${k}:${vals.sort().join("|")}`;
    })
    .sort((a, b) => a.localeCompare(b));
  return `${contentType}|${parts.join(";")}`;
}

function buildContextKey(
  piece: DatasetVideo["pieces"][number],
  facetSubset: Facet[]
): string {
  if (facetSubset.length === 0) return "__NO_FACETS__";
  const labels = (piece.labels ?? {}) as Record<string, string[]>;
  const parts = facetSubset
    .map((f) => {
      const k = facetToLabelKey(f);
      const raw = (labels[k] ?? []) as string[];
      const vals = normalizeFacetVals(raw);
      return [k, vals.sort().join("|")] as const;
    })
    .sort((a, b) => a[0].localeCompare(b[0]));
  return parts.map(([k, v]) => `${k}=${v}`).join(";");
}

function buildReadableContext(contextKey: string, facetSubset: Facet[]): string {
  if (contextKey === "__NO_FACETS__") return "(no facets)";
  return contextKey
    .split(";")
    .map((part) => {
      const eq = part.indexOf("=");
      if (eq < 0) return part;
      const key = part.slice(0, eq);
      const val = part.slice(eq + 1);
      const facet = facetSubset.find((f) => facetToLabelKey(f) === key);
      return `${facet ? facet.title : key.slice(0, 12)}: ${val}`;
    })
    .join(" · ");
}

// ---------------------------------------------------------------------------
// Core metrics (matching the Python reference implementation)
// ---------------------------------------------------------------------------

/**
 * Compactness C = sum of vocabulary sizes (excluding "na" labels).
 * Pure function of the facet schema, no data needed.
 */
export function computeCompactness(facetSubset: Facet[]): number {
  let total = 0;
  for (const facet of facetSubset) {
    for (const v of facet.vocabulary) {
      if (v.label.toLowerCase() !== "na") total += 1;
    }
  }
  return total;
}

/** Build cell → Set<unit_id> mapping for a given facet subset. */
function computeCells(
  allPieces: DatasetVideo["pieces"],
  facetSubset: Facet[]
): Map<string, Set<string>> {
  const cellToUnits = new Map<string, Set<string>>();
  for (const piece of allPieces) {
    if (!piece.unit_id) continue;
    const key = buildCellKey(piece, facetSubset);
    let s = cellToUnits.get(key);
    if (!s) {
      s = new Set();
      cellToUnits.set(key, s);
    }
    s.add(piece.unit_id);
  }
  return cellToUnits;
}

/**
 * Discriminativeness D = Σ_c (n_c / N) · log₂(n_c).
 * D = 0   → perfect separation (each cell contains exactly one unit).
 * D = log₂(N) → no separation (all units in one cell).
 * Lower D is better.
 */
function computeD(
  cellToUnits: Map<string, Set<string>>,
  numUnits: number
): number {
  if (numUnits === 0) return 0;
  let total = 0;
  for (const unitSet of cellToUnits.values()) {
    const n_c = unitSet.size;
    if (n_c > 1) {
      total += (n_c / numUnits) * Math.log2(n_c);
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// 0/1 Knapsack DP to find Pareto-optimal (compactness, discriminativeness) points
//
// Since compactness is additive (sum of per-facet vocab sizes), we use a DP
// table keyed by compactness value.  For each achievable C we keep the facet
// subset with minimum D.  The resulting points are then filtered to the global
// Pareto front (minimize both C and D).
// ---------------------------------------------------------------------------

export function computeDatasetParetoPoints(
  datasetLabel: string,
  dataset: DatasetVideo[],
  allFacets: Facet[]
): PlotPoint[] {
  const allPieces = dataset.flatMap((v) => v.pieces ?? []);
  const numUniqueUnits = new Set(
    allPieces.filter((p) => p.unit_id).map((p) => p.unit_id as string)
  ).size;

  if (allFacets.length === 0 || numUniqueUnits === 0 || allPieces.length === 0) return [];

  type DpEntry = { subset: Facet[]; D: number; cellCount: number };

  // Initialise with the empty subset (compactness = 0)
  const dp = new Map<number, DpEntry>();
  const baseCells = computeCells(allPieces, []);
  dp.set(0, { subset: [], D: computeD(baseCells, numUniqueUnits), cellCount: baseCells.size });

  // Process each facet exactly once (0/1 knapsack)
  for (const facet of allFacets) {
    const facetC = computeCompactness([facet]);
    // Snapshot current dp so we don't reuse the same facet in a single pass
    const snapshot = Array.from(dp.entries());
    for (const [c, entry] of snapshot) {
      const newC = c + facetC;
      const newSubset = [...entry.subset, facet];
      const newCells = computeCells(allPieces, newSubset);
      const newD = computeD(newCells, numUniqueUnits);
      const existing = dp.get(newC);
      if (!existing || newD < existing.D) {
        dp.set(newC, { subset: newSubset, D: newD, cellCount: newCells.size });
      }
    }
  }

  // Collect points (skip empty subset at c = 0)
  const points: PlotPoint[] = [];
  for (const [c, entry] of dp.entries()) {
    if (c === 0) continue;
    points.push({
      datasetLabel,
      facetSubset: entry.subset,
      compactness: c,
      discriminativeness: entry.D,
      relativeTotal: entry.cellCount,
      numUniqueUnits,
    });
  }

  // Filter to Pareto front within this dataset (eliminates any remaining dominated points)
  const paretoIdx = computeParetoFrontIndices(points);
  return points.filter((_, i) => paretoIdx.has(i));
}

/**
 * Compute Pareto-front indices over an array of PlotPoints.
 * Minimises both compactness and discriminativeness.
 * A point p is dominated if there exists q with q.C ≤ p.C AND q.D ≤ p.D
 * (at least one strictly less).
 */
export function computeParetoFrontIndices(points: PlotPoint[]): Set<number> {
  const pareto = new Set<number>();
  for (let i = 0; i < points.length; i++) {
    let dominated = false;
    const p = points[i];
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j];
      if (
        q.compactness <= p.compactness &&
        q.discriminativeness <= p.discriminativeness &&
        (q.compactness < p.compactness || q.discriminativeness < p.discriminativeness)
      ) {
        dominated = true;
        break;
      }
    }
    if (!dominated) pareto.add(i);
  }
  return pareto;
}

// ---------------------------------------------------------------------------
// InspectionPanel helpers
// ---------------------------------------------------------------------------

/** For each facet in the subset, count pieces per vocabulary value. */
export function computeFacetDistributions(
  dataset: DatasetVideo[],
  facetSubset: Facet[]
): FacetValueDist[] {
  const allPieces = dataset.flatMap((v) => v.pieces ?? []);
  return facetSubset.map((facet) => {
    const key = facetToLabelKey(facet);
    const countMap = new Map<string, number>();
    for (const piece of allPieces) {
      const raw = (piece.labels?.[key] ?? []) as string[];
      const vals = normalizeFacetVals(raw);
      for (const val of vals) {
        countMap.set(val, (countMap.get(val) ?? 0) + 1);
      }
    }
    const total = Array.from(countMap.values()).reduce((s, c) => s + c, 0);
    const valueCounts = facet.vocabulary.map((v) => ({
      label: v.label,
      count: countMap.get(v.label) ?? 0,
      percentage: total > 0 ? ((countMap.get(v.label) ?? 0) / total) * 100 : 0,
    }));
    const naCount = countMap.get("na") ?? 0;
    if (naCount > 0) {
      valueCounts.push({
        label: "na",
        count: naCount,
        percentage: total > 0 ? (naCount / total) * 100 : 0,
      });
    }
    return { facet, valueCounts, totalPieces: allPieces.length };
  });
}

/** Per-video context-breakdown under the selected facet subset. */
export function computeVideoBreakdowns(
  dataset: DatasetVideo[],
  facetSubset: Facet[]
): VideoBreakdown[] {
  return dataset.map((video) => {
    const pieces = video.pieces ?? [];
    const contextMap = new Map<string, number>();
    for (const piece of pieces) {
      const key = buildContextKey(piece, facetSubset);
      contextMap.set(key, (contextMap.get(key) ?? 0) + 1);
    }
    return {
      videoId: video.id ?? video.url,
      title: video.title,
      totalPieces: pieces.length,
      uniqueContexts: contextMap.size,
      contextCounts: Array.from(contextMap.entries())
        .map(([context, count]) => ({
          context,
          readableContext: buildReadableContext(context, facetSubset),
          count,
        }))
        .sort((a, b) => b.count - a.count),
    };
  });
}
