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

export type VideoChapter = {
  startTime: number;
  endTime: number;
  context: string;
  readableContext: string;
  pieceCount: number;
};

export type VideoBreakdown = {
  videoId: string;
  title: string;
  videoUrl: string;
  totalPieces: number;
  uniqueContexts: number;
  contextCounts: { context: string; readableContext: string; count: number }[];
  chapters: VideoChapter[];
};

export type CellCount = {
  cellKey: string;
  contentType: string;
  readableContext: string;
  count: number;
  percentage: number;
  /** Up to 3 example piece content strings for tooltip preview. */
  sampleContent: string[];
};

export type PieceInfo = {
  pieceId: string;
  content: string;
  start: number;
  end: number;
  videoId: string;
  videoTitle: string;
};

export type UnitGroup = {
  unitId: string;
  pieces: PieceInfo[];
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

export function buildContextKey(
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

export function buildReadableContext(contextKey: string, facetSubset: Facet[]): string {
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
// Exhaustive subset enumeration to find Pareto-optimal
// (compactness, discriminativeness) points.
//
// All subsets up to MAX_FACETS_CAP are pre-computed once on dataset load and
// stored as a CandidateMap keyed by (compactness, facetCount).
// computeParetoFromCandidates then derives the Pareto front for any
// maxFacets ≤ MAX_FACETS_CAP instantly — no async recomputation needed.
// ---------------------------------------------------------------------------

export const MAX_FACETS_CAP = 3;

export type CandidateEntry = {
  D: number;
  facetIndices: number[];
  cellCount: number;
  k: number; // facet count
  c: number; // compactness
};

/** Keyed by `"${compactness},${facetCount}"` */
export type DatasetCandidates = {
  datasetLabel: string;
  ckMap: Map<string, CandidateEntry>;
  sortedFacets: Facet[];
  numUniqueUnits: number;
};

/** Push all k-combinations of indices [0..n-1] into `out`. */
function generateCombinations(n: number, k: number, out: number[][]): void {
  const current: number[] = [];
  function recurse(start: number): void {
    if (current.length === k) { out.push([...current]); return; }
    const remaining = k - current.length;
    for (let i = start; i <= n - remaining; i++) {
      current.push(i);
      recurse(i + 1);
      current.pop();
    }
  }
  recurse(0);
}

/**
 * Enumerate all subsets up to MAX_FACETS_CAP and store the best (min-D)
 * entry per (compactness, facetCount) bucket.  Returns null when the
 * dataset has no usable pieces.
 */
export async function computeDatasetCandidates(
  datasetLabel: string,
  dataset: DatasetVideo[],
  allFacets: Facet[],
  infoTypes: string[] | null,
  onProgress?: (done: number, total: number) => void
): Promise<DatasetCandidates | null> {
  const rawPieces = dataset.flatMap((v) => v.pieces ?? []);
  const allPieces =
    infoTypes && infoTypes.length > 0
      ? rawPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
      : rawPieces;

  // Sort facets by key name for deterministic, consistent component ordering
  const sortedFacets = [...allFacets].sort((a, b) =>
    facetToLabelKey(a).localeCompare(facetToLabelKey(b))
  );
  const facetKeys = sortedFacets.map(facetToLabelKey);

  // Pre-compute per-piece data once: content_type + per-facet label strings
  // This avoids redundant normalizeFacetVals / sort / join calls in the inner loop
  type PieceData = { contentType: string; components: string[]; unitId: string };
  const validPieces: PieceData[] = [];
  const unitIdSet = new Set<string>();

  for (const piece of allPieces) {
    if (!piece.unit_id) continue;
    unitIdSet.add(piece.unit_id as string);
    const labels = (piece.labels ?? {}) as Record<string, string[]>;
    const components = facetKeys.map((k) => {
      const raw = (labels[k] ?? []) as string[];
      const vals = normalizeFacetVals(raw);
      return vals.sort().join("|");
    });
    validPieces.push({
      contentType: piece.content_type ?? "unknown",
      components,
      unitId: piece.unit_id as string,
    });
  }

  const numUniqueUnits = unitIdSet.size;
  if (sortedFacets.length === 0 || numUniqueUnits === 0 || validPieces.length === 0) return null;

  // Build cell key from pre-computed data — facetIndices are always in ascending order
  function buildKeyFast(pc: PieceData, facetIndices: number[]): string {
    if (facetIndices.length === 0) return pc.contentType;
    const parts = facetIndices.map((fi) => `${facetKeys[fi]}:${pc.components[fi]}`);
    return `${pc.contentType}|${parts.join(";")}`;
  }

  function computeCellsFast(facetIndices: number[]): Map<string, Set<string>> {
    const cellToUnits = new Map<string, Set<string>>();
    for (const pc of validPieces) {
      const key = buildKeyFast(pc, facetIndices);
      let s = cellToUnits.get(key);
      if (!s) { s = new Set(); cellToUnits.set(key, s); }
      s.add(pc.unitId);
    }
    return cellToUnits;
  }

  // Memoised evaluation — each unique subset is computed at most once
  const evalMemo = new Map<string, { D: number; cellCount: number }>();
  const evalSubset = (facetIndices: number[]): { D: number; cellCount: number } => {
    const key = facetIndices.join(',');
    const cached = evalMemo.get(key);
    if (cached) return cached;
    const cells = computeCellsFast(facetIndices);
    const result = { D: computeD(cells, numUniqueUnits), cellCount: cells.size };
    evalMemo.set(key, result);
    return result;
  };

  const facetCompactness = sortedFacets.map((f) => computeCompactness([f]));

  // Enumerate subsets of size 0 (empty baseline) through MAX_FACETS_CAP
  const combos: number[][] = [[]];
  for (let k = 1; k <= Math.min(MAX_FACETS_CAP, sortedFacets.length); k++) {
    generateCombinations(sortedFacets.length, k, combos);
  }

  // ckMap[`${c},${k}`] = best (min-D) entry for that (compactness, facetCount) bucket
  const ckMap = new Map<string, CandidateEntry>();
  // Subsets with D=0: adding more facets keeps D=0 at strictly higher C → dominated, skip
  const perfect: number[][] = [];

  const YIELD_EVERY = 200;

  for (let ci = 0; ci < combos.length; ci++) {
    const facetIndices = combos[ci];

    // Monotonicity pruning: skip if any already-found D=0 subset is ⊆ this combo
    if (perfect.length > 0) {
      const comboSet = new Set(facetIndices);
      if (perfect.some((p) => p.every((i) => comboSet.has(i)))) {
        if ((ci + 1) % YIELD_EVERY === 0) {
          onProgress?.(ci + 1, combos.length);
          await new Promise<void>((r) => setTimeout(r, 0));
        }
        continue;
      }
    }

    const c = facetIndices.reduce((s, fi) => s + facetCompactness[fi], 0);
    const k = facetIndices.length;
    const { D, cellCount } = evalSubset(facetIndices);

    const ckKey = `${c},${k}`;
    const existing = ckMap.get(ckKey);
    if (!existing || D < existing.D) {
      ckMap.set(ckKey, { D, facetIndices: [...facetIndices], cellCount, k, c });
    }
    if (D === 0) perfect.push([...facetIndices]);

    if ((ci + 1) % YIELD_EVERY === 0) {
      onProgress?.(ci + 1, combos.length);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }
  onProgress?.(combos.length, combos.length);

  return { datasetLabel, ckMap, sortedFacets, numUniqueUnits };
}

/**
 * Instant (synchronous) — filter pre-computed candidates to subsets of
 * size ≤ maxFacets, pick the min-D winner per compactness value, and
 * return the Pareto front sorted by ascending compactness.
 */
export function computeParetoFromCandidates(
  candidates: DatasetCandidates,
  maxFacets: number
): PlotPoint[] {
  // For each compactness C, find the min-D entry among subsets with k ≤ maxFacets
  const bestPerC = new Map<number, CandidateEntry>();
  for (const entry of candidates.ckMap.values()) {
    if (entry.k > maxFacets) continue;
    const existing = bestPerC.get(entry.c);
    if (!existing || entry.D < existing.D) bestPerC.set(entry.c, entry);
  }

  const raw = Array.from(bestPerC.entries())
    .map(([c, entry]) => ({
      datasetLabel: candidates.datasetLabel,
      facetSubset: entry.facetIndices.map((fi) => candidates.sortedFacets[fi]),
      compactness: c,
      discriminativeness: entry.D,
      relativeTotal: entry.cellCount,
      numUniqueUnits: candidates.numUniqueUnits,
    }))
    .sort((a, b) => a.compactness - b.compactness);

  // Pareto filter: keep only points where D strictly decreases as C increases
  const smoothed: PlotPoint[] = [];
  for (const point of raw) {
    if (smoothed.length > 0 && smoothed[smoothed.length - 1].discriminativeness <= point.discriminativeness) continue;
    smoothed.push(point);
  }
  return smoothed;
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
  facetSubset: Facet[],
  infoTypes: string[] | null
): FacetValueDist[] {
  const rawPieces = dataset.flatMap((v) => v.pieces ?? []);
  const allPieces =
    infoTypes && infoTypes.length > 0
      ? rawPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
      : rawPieces;
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
  facetSubset: Facet[],
  infoTypes: string[] | null
): VideoBreakdown[] {
  return dataset.map((video) => {
    const rawPieces = video.pieces ?? [];
    const pieces =
      infoTypes && infoTypes.length > 0
        ? rawPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
        : rawPieces;
    const contextMap = new Map<string, number>();
    for (const piece of pieces) {
      const key = buildContextKey(piece, facetSubset);
      contextMap.set(key, (contextMap.get(key) ?? 0) + 1);
    }

    // Build time-ordered chapters (consecutive runs of same context)
    const sorted = [...pieces].sort((a, b) => a.start - b.start);
    const chapters: VideoChapter[] = [];
    for (const piece of sorted) {
      const ctx = buildContextKey(piece, facetSubset);
      const last = chapters[chapters.length - 1];
      if (last && last.context === ctx) {
        last.endTime = piece.end;
        last.pieceCount++;
      } else {
        chapters.push({
          startTime: piece.start,
          endTime: piece.end,
          context: ctx,
          readableContext: buildReadableContext(ctx, facetSubset),
          pieceCount: 1,
        });
      }
    }

    return {
      videoId: video.id ?? video.url,
      title: video.title,
      videoUrl: video.url,
      totalPieces: pieces.length,
      uniqueContexts: contextMap.size,
      contextCounts: Array.from(contextMap.entries())
        .map(([context, count]) => ({
          context,
          readableContext: buildReadableContext(context, facetSubset),
          count,
        }))
        .sort((a, b) => b.count - a.count),
      chapters,
    };
  });
}

/** Top-N cells (content_type × facet-label combination) by piece count across all videos. */
export function computeTopCells(
  dataset: DatasetVideo[],
  facetSubset: Facet[],
  topN: number,
  infoTypes: string[] | null
): CellCount[] {
  const rawPieces = dataset.flatMap((v) => v.pieces ?? []);
  const allPieces =
    infoTypes && infoTypes.length > 0
      ? rawPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
      : rawPieces;
  const cellMap = new Map<string, {
    contentType: string;
    contextKey: string;
    count: number;
    samples: string[];
  }>();

  for (const piece of allPieces) {
    const contentType = piece.content_type ?? "unknown";
    const contextKey = buildContextKey(piece, facetSubset);
    const cellKey = `${contentType}|||${contextKey}`;
    const existing = cellMap.get(cellKey);
    if (existing) {
      existing.count++;
      if (existing.samples.length < 3) existing.samples.push(piece.content);
    } else {
      cellMap.set(cellKey, { contentType, contextKey, count: 1, samples: [piece.content] });
    }
  }

  const total = allPieces.length;
  return Array.from(cellMap.entries())
    .map(([cellKey, { contentType, contextKey, count, samples }]) => ({
      cellKey,
      contentType,
      readableContext: buildReadableContext(contextKey, facetSubset),
      count,
      percentage: total > 0 ? (count / total) * 100 : 0,
      sampleContent: samples,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, topN);
}

/** All pieces in a given cell (content_type × facet-label key), grouped by unit_id. */
export function computeCellPieces(
  dataset: DatasetVideo[],
  facetSubset: Facet[],
  targetCellKey: string,
  infoTypes: string[] | null
): UnitGroup[] {
  const unitMap = new Map<string, PieceInfo[]>();
  const isIncluded = (ct: string) =>
    !infoTypes || infoTypes.length === 0 || infoTypes.includes(ct);

  for (const video of dataset) {
    const videoId = video.id ?? video.url;
    for (const piece of video.pieces ?? []) {
      const contentType = piece.content_type ?? "unknown";
      if (!isIncluded(contentType)) continue;
      const contextKey = buildContextKey(piece, facetSubset);
      if (`${contentType}|||${contextKey}` !== targetCellKey) continue;

      const info: PieceInfo = {
        pieceId: piece.piece_id,
        content: piece.content,
        start: piece.start,
        end: piece.end,
        videoId,
        videoTitle: video.title,
      };
      const existing = unitMap.get(piece.unit_id);
      if (existing) {
        existing.push(info);
      } else {
        unitMap.set(piece.unit_id, [info]);
      }
    }
  }

  return Array.from(unitMap.entries())
    .map(([unitId, pieces]) => ({
      unitId,
      pieces: pieces.sort((a, b) => a.start - b.start),
    }))
    .sort((a, b) => (a.pieces[0]?.start ?? 0) - (b.pieces[0]?.start ?? 0));
}
