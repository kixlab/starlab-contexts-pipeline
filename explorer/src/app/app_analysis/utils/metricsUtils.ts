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

/** Synthetic facet ID for content_type (treated as a pinnable facet). */
export const CONTENT_TYPE_FACET_ID = "__content_type__";

/**
 * Build a synthetic Facet representing piece.content_type values.
 * Pinned by default when the dataset has an explicit infoTypes list.
 */
export function buildContentTypeFacet(infoTypes: string[]): Facet {
  return {
    id: CONTENT_TYPE_FACET_ID,
    type: "subtype",
    title: "Content Type",
    title_plural: "Content Types",
    definition: "The subtype (content_type field) of each information piece.",
    vocabulary: infoTypes.map((ct) => ({ label: ct, definition: ct })),
  };
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
      // Special handling for synthetic content_type facet
      const raw: string[] =
        k === CONTENT_TYPE_FACET_ID
          ? [piece.content_type ?? "unknown"]
          : ((labels[k] ?? []) as string[]);
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
// All subsets of FREE facets up to MAX_FACETS_CAP are pre-computed once
// (given a set of pinned facets that are always included). Pinning changes
// the search space so computeDatasetCandidates must be re-run when pins change.
//
// computeParetoFromCandidates then derives the Pareto front for any
// maxFacets ≤ MAX_FACETS_CAP instantly — no async recomputation needed.
// ---------------------------------------------------------------------------

export const MAX_FACETS_CAP = 3;

export type CandidateEntry = {
  D: number;
  facetIndices: number[]; // indices into DatasetCandidates.sortedFacets (FREE facets only)
  cellCount: number;
  k: number; // free facet count
  c: number; // total compactness (pinned + free)
};

/** Keyed by `"${compactness},${freeK}"` */
export type DatasetCandidates = {
  datasetLabel: string;
  ckMap: Map<string, CandidateEntry>;
  /** FREE facets only (not pinned). Indices in CandidateEntry refer to this array. */
  sortedFacets: Facet[];
  /** Always included in every point's facetSubset. */
  pinnedFacets: Facet[];
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
 * Enumerate all FREE subsets up to MAX_FACETS_CAP and store the best (min-D)
 * entry per (compactness, freeK) bucket. Pinned facets are always included in
 * every evaluated subset (affecting D and C) but are not permuted.
 *
 * Returns null when the dataset has no usable pieces.
 */
export async function computeDatasetCandidates(
  datasetLabel: string,
  dataset: DatasetVideo[],
  allFacets: Facet[],
  infoTypes: string[] | null,
  pinnedFacetIds: Set<string>,
  onProgress?: (done: number, total: number) => void
): Promise<DatasetCandidates | null> {
  const rawPieces = dataset.flatMap((v) => v.pieces ?? []);
  const allPieces =
    infoTypes && infoTypes.length > 0
      ? rawPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
      : rawPieces;

  // Sort all facets for deterministic ordering, then split pinned vs free
  const allSorted = [...allFacets].sort((a, b) =>
    facetToLabelKey(a).localeCompare(facetToLabelKey(b))
  );
  const pinnedFacets = allSorted.filter((f) => pinnedFacetIds.has(f.id));
  const freeFacets = allSorted.filter((f) => !pinnedFacetIds.has(f.id));

  const pinnedKeys = pinnedFacets.map(facetToLabelKey);
  const freeKeys = freeFacets.map(facetToLabelKey);

  // Pre-compute per-piece data once: per-facet label strings for both pinned and free
  type PieceData = {
    contentType: string;
    pinnedComponents: string[];
    freeComponents: string[];
    unitId: string;
  };
  const validPieces: PieceData[] = [];
  const unitIdSet = new Set<string>();

  for (const piece of allPieces) {
    if (!piece.unit_id) continue;
    unitIdSet.add(piece.unit_id as string);
    const labels = (piece.labels ?? {}) as Record<string, string[]>;

    const getComponent = (k: string): string => {
      if (k === CONTENT_TYPE_FACET_ID) return piece.content_type ?? "unknown";
      const raw = (labels[k] ?? []) as string[];
      return normalizeFacetVals(raw).sort().join("|");
    };

    const pinnedComponents = pinnedKeys.map(getComponent);
    const freeComponents = freeKeys.map(getComponent);

    validPieces.push({
      contentType: piece.content_type ?? "unknown",
      pinnedComponents,
      freeComponents,
      unitId: piece.unit_id as string,
    });
  }

  const numUniqueUnits = unitIdSet.size;
  // We need at least pinned facets; can still proceed with 0 free facets
  if (numUniqueUnits === 0 || validPieces.length === 0) return null;

  // Compactness of pinned facets (fixed cost in every subset)
  const pinnedCompactness = computeCompactness(pinnedFacets);
  const freeCompactness = freeFacets.map((f) => computeCompactness([f]));

  /**
   * Build a cell key that is unique per (pinned context) + (free context subset).
   * The content_type prefix is omitted when __content_type__ is already encoded
   * as a facet component (either pinned or free) to avoid double-encoding.
   */
  const contentTypePinned = pinnedFacetIds.has(CONTENT_TYPE_FACET_ID);

  function buildKeyFast(pc: PieceData, freeIndices: number[]): string {
    const contentTypeFree =
      !contentTypePinned && freeIndices.some((fi) => freeKeys[fi] === CONTENT_TYPE_FACET_ID);
    const hasContentTypeFacet = contentTypePinned || contentTypeFree;

    // Build pinned parts
    const pinnedParts = pinnedKeys.map((k, i) => `${k}:${pc.pinnedComponents[i]}`);
    // Build free parts from selected indices
    const freeParts = freeIndices.map((fi) => `${freeKeys[fi]}:${pc.freeComponents[fi]}`);

    const allParts = [...pinnedParts, ...freeParts].sort();

    if (allParts.length === 0) {
      // No facets at all — use content_type as sole discriminator
      return pc.contentType;
    }

    const prefix = hasContentTypeFacet ? "" : `${pc.contentType}|`;
    return `${prefix}${allParts.join(";")}`;
  }

  function computeCellsFast(freeIndices: number[]): Map<string, Set<string>> {
    const cellToUnits = new Map<string, Set<string>>();
    for (const pc of validPieces) {
      const key = buildKeyFast(pc, freeIndices);
      let s = cellToUnits.get(key);
      if (!s) { s = new Set(); cellToUnits.set(key, s); }
      s.add(pc.unitId);
    }
    return cellToUnits;
  }

  // Memoised evaluation — each unique free-subset is computed at most once
  const evalMemo = new Map<string, { D: number; cellCount: number }>();
  const evalSubset = (freeIndices: number[]): { D: number; cellCount: number } => {
    const key = freeIndices.join(",");
    const cached = evalMemo.get(key);
    if (cached) return cached;
    const cells = computeCellsFast(freeIndices);
    const result = { D: computeD(cells, numUniqueUnits), cellCount: cells.size };
    evalMemo.set(key, result);
    return result;
  };

  // Enumerate subsets of FREE facets of size 0 through MAX_FACETS_CAP
  const combos: number[][] = [[]];
  for (let k = 1; k <= Math.min(MAX_FACETS_CAP, freeFacets.length); k++) {
    generateCombinations(freeFacets.length, k, combos);
  }

  // ckMap[`${totalC},${freeK}`] = best (min-D) entry for that bucket
  const ckMap = new Map<string, CandidateEntry>();
  // Subsets with D=0: adding more free facets keeps D=0 at strictly higher C → dominated, skip
  const perfect: number[][] = [];

  const YIELD_EVERY = 200;

  for (let ci = 0; ci < combos.length; ci++) {
    const freeIndices = combos[ci];

    // Monotonicity pruning: skip if any already-found D=0 free subset is ⊆ this combo
    if (perfect.length > 0) {
      const comboSet = new Set(freeIndices);
      if (perfect.some((p) => p.every((i) => comboSet.has(i)))) {
        if ((ci + 1) % YIELD_EVERY === 0) {
          onProgress?.(ci + 1, combos.length);
          await new Promise<void>((r) => setTimeout(r, 0));
        }
        continue;
      }
    }

    const freeC = freeIndices.reduce((s, fi) => s + freeCompactness[fi], 0);
    const totalC = pinnedCompactness + freeC;
    const k = freeIndices.length;
    const { D, cellCount } = evalSubset(freeIndices);

    const ckKey = `${totalC},${k}`;
    const existing = ckMap.get(ckKey);
    if (!existing || D < existing.D) {
      ckMap.set(ckKey, { D, facetIndices: [...freeIndices], cellCount, k, c: totalC });
    }
    if (D === 0) perfect.push([...freeIndices]);

    if ((ci + 1) % YIELD_EVERY === 0) {
      onProgress?.(ci + 1, combos.length);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }
  onProgress?.(combos.length, combos.length);

  return { datasetLabel, ckMap, sortedFacets: freeFacets, pinnedFacets, numUniqueUnits };
}

/**
 * Instant (synchronous) — filter pre-computed candidates to free subsets of
 * size ≤ maxFacets, pick the min-D winner per total compactness value, and
 * return the Pareto front sorted by ascending compactness.
 *
 * Each point's facetSubset includes both pinned and free facets.
 */
export function computeParetoFromCandidates(
  candidates: DatasetCandidates,
  maxFacets: number
): PlotPoint[] {
  // For each total compactness C, find the min-D entry among free subsets with k ≤ maxFacets
  const bestPerC = new Map<number, CandidateEntry>();
  for (const entry of candidates.ckMap.values()) {
    if (entry.k > maxFacets) continue;
    const existing = bestPerC.get(entry.c);
    if (!existing || entry.D < existing.D) bestPerC.set(entry.c, entry);
  }

  const raw = Array.from(bestPerC.entries())
    .map(([c, entry]) => ({
      datasetLabel: candidates.datasetLabel,
      // Full facetSubset = pinned + free
      facetSubset: [
        ...candidates.pinnedFacets,
        ...entry.facetIndices.map((fi) => candidates.sortedFacets[fi]),
      ],
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
      let raw: string[];
      if (key === CONTENT_TYPE_FACET_ID) {
        raw = [piece.content_type ?? "unknown"];
      } else {
        raw = (piece.labels?.[key] ?? []) as string[];
      }
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

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

export type CellBreakdownEntry = {
  readableContext: string;
  count: number;
  percentage: number;
};

/**
 * For a given anchor cell, show how a comparison schema breaks it down.
 * Returns the distribution of comparison-schema context keys among pieces
 * that belong to the anchor cell.
 */
export function computeCellBreakdown(
  dataset: DatasetVideo[],
  anchorFacets: Facet[],
  comparisonFacets: Facet[],
  anchorCellKey: string,
  infoTypes: string[] | null
): CellBreakdownEntry[] {
  const isIncluded = (ct: string) =>
    !infoTypes || infoTypes.length === 0 || infoTypes.includes(ct);

  const countMap = new Map<string, number>();
  let total = 0;

  for (const video of dataset) {
    for (const piece of video.pieces ?? []) {
      const contentType = piece.content_type ?? "unknown";
      if (!isIncluded(contentType)) continue;
      const aKey = buildContextKey(piece, anchorFacets);
      const aCellKey = `${contentType}|||${aKey}`;
      if (aCellKey !== anchorCellKey) continue;
      total++;
      const cKey = buildContextKey(piece, comparisonFacets);
      const readable = buildReadableContext(cKey, comparisonFacets);
      countMap.set(readable, (countMap.get(readable) ?? 0) + 1);
    }
  }

  return Array.from(countMap.entries())
    .map(([readableContext, count]) => ({
      readableContext,
      count,
      percentage: total > 0 ? (count / total) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count);
}

export type MergedChapterEntry = {
  startTime: number;
  endTime: number;
  context0: string;
  context1: string;
  readableContext0: string;
  readableContext1: string;
  pieceCount0: number;
  pieceCount1: number;
};

/**
 * Merge chapters from two schemas for the same video into a unified timeline.
 * Each entry spans a time interval where both schemas have a constant context.
 * The result is sorted by startTime.
 */
export function mergeVideoChapters(
  chapters0: VideoChapter[],
  chapters1: VideoChapter[]
): MergedChapterEntry[] {
  // Collect all unique boundary times from both schemas
  const boundaries = new Set<number>();
  for (const ch of chapters0) { boundaries.add(ch.startTime); boundaries.add(ch.endTime); }
  for (const ch of chapters1) { boundaries.add(ch.startTime); boundaries.add(ch.endTime); }
  const times = Array.from(boundaries).sort((a, b) => a - b);

  if (times.length < 2) return [];

  // For each interval [times[i], times[i+1]), find the active chapter in each schema
  const findChapter = (chapters: VideoChapter[], t: number) =>
    chapters.find((ch) => ch.startTime <= t && ch.endTime > t) ?? null;

  const merged: MergedChapterEntry[] = [];
  for (let i = 0; i < times.length - 1; i++) {
    const start = times[i];
    const end = times[i + 1];
    const mid = (start + end) / 2;
    const ch0 = findChapter(chapters0, mid);
    const ch1 = findChapter(chapters1, mid);
    if (!ch0 && !ch1) continue;
    merged.push({
      startTime: start,
      endTime: end,
      context0: ch0?.context ?? "",
      context1: ch1?.context ?? "",
      readableContext0: ch0?.readableContext ?? "—",
      readableContext1: ch1?.readableContext ?? "—",
      pieceCount0: ch0?.pieceCount ?? 0,
      pieceCount1: ch1?.pieceCount ?? 0,
    });
  }

  // Merge consecutive entries with identical context pair
  const compressed: MergedChapterEntry[] = [];
  for (const entry of merged) {
    const last = compressed[compressed.length - 1];
    if (last && last.context0 === entry.context0 && last.context1 === entry.context1) {
      last.endTime = entry.endTime;
      last.pieceCount0 += entry.pieceCount0;
      last.pieceCount1 += entry.pieceCount1;
    } else {
      compressed.push({ ...entry });
    }
  }

  return compressed;
}
