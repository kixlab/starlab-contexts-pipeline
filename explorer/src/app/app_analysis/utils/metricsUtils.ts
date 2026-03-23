import type { Facet, DatasetVideo } from "@/app/types";
import { facetToLabelKey, normalizeFacetVals, computeD } from "@/app/utils/facetStats";

/** Which metric to use on the Y axis of the scatter plot. */
export type YMetricType = "discriminativeness" | "etaSquared";

/** Population variance of a numeric array. */
function computeVariance(values: number[]): number {
  if (values.length <= 1) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length;
}

export type PlotPoint = {
  datasetLabel: string;
  facetSubset: Facet[];
  /** Sum of vocabulary sizes (excl. "na") — lower = more compact */
  compactness: number;
  /** D = Σ_c (n_c/N)·log₂(n_c) — lower = more discriminative (0 = perfect separation) */
  discriminativeness: number;
  /** η² = 1 - SS_within/SS_total — higher = signatures better predict temporal position (1 = perfect) */
  eta2: number;
  /** 1 - avg transition rate across tutorials. Higher = facet values change less often between consecutive items. */
  stability: number;
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
  eta2: number;
  stability: number;
  facetIndices: number[]; // indices into DatasetCandidates.sortedFacets (FREE facets only)
  cellCount: number;
  k: number; // free facet count
  c: number; // total compactness (pinned + free)
};

/** Keyed by `"${compactness},${freeK}"` */
export type DatasetCandidates = {
  datasetLabel: string;
  /** Best min-D entry per (C, k) bucket */
  ckMap: Map<string, CandidateEntry>;
  /** Best max-η² entry per (C, k) bucket */
  ckMapEta2: Map<string, CandidateEntry>;
  /** FREE facets only (not included/excluded). Indices in CandidateEntry refer to this array. */
  sortedFacets: Facet[];
  /** Always included in every point's facetSubset. */
  includedFacets: Facet[];
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
  includedFacetIds: Set<string>,
  excludedFacetIds: Set<string>,
  onProgress?: (done: number, total: number) => void
): Promise<DatasetCandidates | null> {
  // Sort all facets for deterministic ordering, filter out excluded, then split included vs free
  const allSorted = [...allFacets]
    .filter((f) => !excludedFacetIds.has(f.id))
    .sort((a, b) => facetToLabelKey(a).localeCompare(facetToLabelKey(b)));
  const pinnedFacets = allSorted.filter((f) => includedFacetIds.has(f.id));
  const freeFacets = allSorted.filter((f) => !includedFacetIds.has(f.id));

  const pinnedKeys = pinnedFacets.map(facetToLabelKey);
  const freeKeys = freeFacets.map(facetToLabelKey);

  // Pre-compute per-piece data once: per-facet label strings for both pinned and free.
  // Iterate per video to preserve video identity and temporal position for η².
  type PieceData = {
    contentType: string;
    pinnedComponents: string[];
    freeComponents: string[];
    unitId: string;
    videoId: string;
    positionInVideo: number;
  };
  const validPieces: PieceData[] = [];
  const unitIdSet = new Set<string>();

  for (const video of dataset) {
    const videoId = video.id ?? video.url;
    const videoPieces = video.pieces ?? [];
    const filtered =
      infoTypes && infoTypes.length > 0
        ? videoPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
        : videoPieces;

    // Sort by start time for temporal position assignment
    const sorted = [...filtered].sort((a, b) => a.start - b.start);

    let posIdx = 0;
    for (const piece of sorted) {
      if (!piece.unit_id) continue;
      unitIdSet.add(piece.unit_id as string);
      const labels = (piece.labels ?? {}) as Record<string, string[]>;

      const getComponent = (k: string): string => {
        if (k === CONTENT_TYPE_FACET_ID) return piece.content_type ?? "unknown";
        const raw = (labels[k] ?? []) as string[];
        return normalizeFacetVals(raw).sort().join("|");
      };

      validPieces.push({
        contentType: piece.content_type ?? "unknown",
        pinnedComponents: pinnedKeys.map(getComponent),
        freeComponents: freeKeys.map(getComponent),
        unitId: piece.unit_id as string,
        videoId,
        positionInVideo: posIdx,
      });
      posIdx++;
    }
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
  const contentTypePinned = includedFacetIds.has(CONTENT_TYPE_FACET_ID);

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

  /**
   * Compute η² (eta-squared) for a facet subset.
   * Groups pieces by video, computes within-video η², returns equal-weight average.
   */
  function computeEta2ForSubset(freeIndices: number[]): number {
    const tutorials = new Map<string, Array<{ position: number; contextKey: string }>>();

    for (const pc of validPieces) {
      const key = buildKeyFast(pc, freeIndices);
      let arr = tutorials.get(pc.videoId);
      if (!arr) { arr = []; tutorials.set(pc.videoId, arr); }
      arr.push({ position: pc.positionInVideo, contextKey: key });
    }

    let sum = 0;
    let count = 0;

    for (const items of tutorials.values()) {
      if (items.length <= 1) continue;

      const allPositions = items.map((it) => it.position);
      const totalVar = computeVariance(allPositions);
      if (totalVar === 0) continue;

      const groups = new Map<string, number[]>();
      for (const item of items) {
        let g = groups.get(item.contextKey);
        if (!g) { g = []; groups.set(item.contextKey, g); }
        g.push(item.position);
      }

      let withinVar = 0;
      for (const positions of groups.values()) {
        withinVar += (positions.length / items.length) * computeVariance(positions);
      }

      sum += 1 - withinVar / totalVar;
      count++;
    }

    return count > 0 ? sum / count : 0;
  }

  /**
   * Compute Stability = 1 - avg transition rate for a facet subset.
   * Transition rate per tutorial = #(consecutive items with different context) / (n - 1).
   * Averaged across tutorials (equal weight).
   */
  function computeStabilityForSubset(freeIndices: number[]): number {
    // Group pieces by video, preserving temporal order
    const tutorials = new Map<string, string[]>(); // videoId → ordered context keys

    for (const pc of validPieces) {
      const key = buildKeyFast(pc, freeIndices);
      let arr = tutorials.get(pc.videoId);
      if (!arr) { arr = []; tutorials.set(pc.videoId, arr); }
      arr.push(key);
    }

    let trSum = 0;
    let trCount = 0;

    for (const contextKeys of tutorials.values()) {
      const n = contextKeys.length;
      if (n <= 1) continue;
      let transitions = 0;
      for (let i = 0; i < n - 1; i++) {
        if (contextKeys[i] !== contextKeys[i + 1]) transitions++;
      }
      trSum += transitions / (n - 1);
      trCount++;
    }

    const avgTR = trCount > 0 ? trSum / trCount : 0;
    return 1 - avgTR;
  }

  // Memoised evaluation — each unique free-subset is computed at most once
  const evalMemo = new Map<string, { D: number; eta2: number; stability: number; cellCount: number }>();
  const evalSubset = (freeIndices: number[]): { D: number; eta2: number; stability: number; cellCount: number } => {
    const key = freeIndices.join(",");
    const cached = evalMemo.get(key);
    if (cached) return cached;
    const cells = computeCellsFast(freeIndices);
    const result = {
      D: computeD(cells, numUniqueUnits),
      eta2: computeEta2ForSubset(freeIndices),
      stability: computeStabilityForSubset(freeIndices),
      cellCount: cells.size,
    };
    evalMemo.set(key, result);
    return result;
  };

  // Enumerate subsets of FREE facets of size 0 through MAX_FACETS_CAP
  const combos: number[][] = [[]];
  for (let k = 1; k <= Math.min(MAX_FACETS_CAP, freeFacets.length); k++) {
    generateCombinations(freeFacets.length, k, combos);
  }

  // Best min-D entry per (C, k) bucket
  const ckMap = new Map<string, CandidateEntry>();
  // Best max-η² entry per (C, k) bucket
  const ckMapEta2 = new Map<string, CandidateEntry>();

  const YIELD_EVERY = 200;

  for (let ci = 0; ci < combos.length; ci++) {
    const freeIndices = combos[ci];

    const freeC = freeIndices.reduce((s, fi) => s + freeCompactness[fi], 0);
    const totalC = pinnedCompactness + freeC;
    const k = freeIndices.length;
    const { D, eta2, stability, cellCount } = evalSubset(freeIndices);

    const ckKey = `${totalC},${k}`;
    const entry: CandidateEntry = { D, eta2, stability, facetIndices: [...freeIndices], cellCount, k, c: totalC };

    // D: keep min
    const existingD = ckMap.get(ckKey);
    if (!existingD || D < existingD.D) ckMap.set(ckKey, entry);

    // η²: keep max
    const existingEta2 = ckMapEta2.get(ckKey);
    if (!existingEta2 || eta2 > existingEta2.eta2) ckMapEta2.set(ckKey, entry);

    if ((ci + 1) % YIELD_EVERY === 0) {
      onProgress?.(ci + 1, combos.length);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
  }
  onProgress?.(combos.length, combos.length);

  return { datasetLabel, ckMap, ckMapEta2, sortedFacets: freeFacets, includedFacets: pinnedFacets, numUniqueUnits };
}

/**
 * Instant (synchronous) — filter pre-computed candidates to free subsets of
 * size ≤ maxFacets, pick the best-metric winner per total compactness value,
 * and return the Pareto front sorted by ascending compactness.
 *
 * Each point's facetSubset includes both included and free facets.
 */
export function computeParetoFromCandidates(
  candidates: DatasetCandidates,
  maxFacets: number,
  yMetric: YMetricType = "discriminativeness",
  stabilityThreshold: number = 0
): PlotPoint[] {
  const useEta2 = yMetric === "etaSquared";
  const ckMap = useEta2 ? candidates.ckMapEta2 : candidates.ckMap;

  const bestPerC = new Map<number, CandidateEntry>();
  for (const entry of ckMap.values()) {
    if (entry.k > maxFacets) continue;
    if (entry.stability < stabilityThreshold) continue;
    const existing = bestPerC.get(entry.c);
    if (!existing) {
      bestPerC.set(entry.c, entry);
    } else if (useEta2 ? entry.eta2 > existing.eta2 : entry.D < existing.D) {
      bestPerC.set(entry.c, entry);
    }
  }

  const raw = Array.from(bestPerC.entries())
    .map(([c, entry]) => ({
      datasetLabel: candidates.datasetLabel,
      facetSubset: [
        ...candidates.includedFacets,
        ...entry.facetIndices.map((fi) => candidates.sortedFacets[fi]),
      ],
      compactness: c,
      discriminativeness: entry.D,
      eta2: entry.eta2,
      stability: entry.stability,
      relativeTotal: entry.cellCount,
      numUniqueUnits: candidates.numUniqueUnits,
    }))
    .sort((a, b) => a.compactness - b.compactness);

  // Pareto filter: keep only points where the metric strictly improves as C increases.
  // D: lower is better → skip if not decreasing. η²: higher is better → skip if not increasing.
  const smoothed: PlotPoint[] = [];
  for (const point of raw) {
    if (smoothed.length > 0) {
      const last = smoothed[smoothed.length - 1];
      if (useEta2 ? last.eta2 >= point.eta2 : last.discriminativeness <= point.discriminativeness) continue;
    }
    smoothed.push(point);
  }
  return smoothed;
}

/**
 * Compute Pareto-front indices over an array of PlotPoints.
 * For D: minimises both compactness and D.
 * For η²: minimises compactness, maximises η².
 */
export function computeParetoFrontIndices(
  points: PlotPoint[],
  yMetric: YMetricType = "discriminativeness"
): Set<number> {
  const useEta2 = yMetric === "etaSquared";
  const pareto = new Set<number>();
  for (let i = 0; i < points.length; i++) {
    let dominated = false;
    const p = points[i];
    const pY = useEta2 ? p.eta2 : p.discriminativeness;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j];
      const qY = useEta2 ? q.eta2 : q.discriminativeness;
      // C: always minimise. Y: minimise D or maximise η².
      const cOk = q.compactness <= p.compactness;
      const yOk = useEta2 ? qY >= pY : qY <= pY;
      const cStrict = q.compactness < p.compactness;
      const yStrict = useEta2 ? qY > pY : qY < pY;
      if (cOk && yOk && (cStrict || yStrict)) {
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
  contextKey: string;
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

  // Map from raw contextKey → count
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
      countMap.set(cKey, (countMap.get(cKey) ?? 0) + 1);
    }
  }

  return Array.from(countMap.entries())
    .map(([contextKey, count]) => ({
      contextKey,
      readableContext: buildReadableContext(contextKey, comparisonFacets),
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

// ---------------------------------------------------------------------------
// Context-chip utilities (shared between DatasetPanel & InspectionPanel)
// ---------------------------------------------------------------------------

/** Palette of visually distinct facet chip colors (used for both panels). */
export const FACET_CHIP_COLORS = [
  "#8B5CF6", // violet
  "#06B6D4", // cyan
  "#F59E0B", // amber
  "#10B981", // emerald
  "#EC4899", // pink
  "#6366F1", // indigo
  "#14B8A6", // teal
  "#F97316", // orange
  "#A855F7", // purple
  "#0EA5E9", // sky
];

/** Deterministic color for a facet ID (hash-based, same ID → same color). */
export function facetIdToChipColor(facetId: string): string {
  let hash = 0;
  for (let i = 0; i < facetId.length; i++) {
    hash = ((hash << 5) - hash + facetId.charCodeAt(i)) | 0;
  }
  return FACET_CHIP_COLORS[Math.abs(hash) % FACET_CHIP_COLORS.length];
}

/** Compute short abbreviations for facets (e.g. "Content Type" → "CT"). */
export function computeFacetAbbreviations(facetSubset: Facet[]): Map<string, string> {
  const abbrevs = new Map<string, string>();
  const used = new Set<string>();
  for (const facet of facetSubset) {
    const words = facet.title.split(/\s+/).filter(Boolean);
    let abbr: string;
    if (words.length > 1) {
      abbr = words.map((w) => w[0]).join("").toUpperCase().slice(0, 3);
    } else {
      abbr = facet.title.slice(0, 2).toUpperCase();
    }
    let candidate = abbr;
    let i = 1;
    while (used.has(candidate)) {
      candidate = abbr + i;
      i++;
    }
    used.add(candidate);
    abbrevs.set(facet.id, candidate);
  }
  return abbrevs;
}

export type ParsedChipEntry = {
  facetId: string;
  facetTitle: string;
  value: string;
};

/**
 * Parse a context key (from buildContextKey) into structured facet-value pairs.
 * Context key format: `facetId=val1|val2;facetId=val1|val2` (sorted by key).
 */
export function parseContextKey(
  contextKey: string,
  facetSubset: Facet[]
): ParsedChipEntry[] {
  if (!contextKey || contextKey === "__NO_FACETS__") return [];
  return contextKey.split(";").map((part) => {
    const eq = part.indexOf("=");
    if (eq < 0) return { facetId: part, facetTitle: part, value: "" };
    const key = part.slice(0, eq);
    const val = part.slice(eq + 1);
    const facet = facetSubset.find((f) => facetToLabelKey(f) === key);
    return {
      facetId: facet?.id ?? key,
      facetTitle: facet?.title ?? key.slice(0, 12),
      value: val,
    };
  });
}

/**
 * Extract the raw context key from a cellKey (format: `contentType|||contextKey`).
 */
export function extractContextKeyFromCellKey(cellKey: string): string {
  const sep = cellKey.indexOf("|||");
  return sep >= 0 ? cellKey.slice(sep + 3) : cellKey;
}
