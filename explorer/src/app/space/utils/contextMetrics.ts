import type { Facet, DatasetVideo } from "@/app/types";
import { facetToLabelKey, normalizeFacetVals } from "@/app/utils/facetStats";

export type FacetValueDist = {
  facet: Facet;
  valueCounts: { label: string; count: number; percentage: number }[];
  totalPieces: number;
};

/** Synthetic facet ID for content_type (treated as a pinnable facet). */
export const CONTENT_TYPE_FACET_ID = "__content_type__";

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

// ---------------------------------------------------------------------------
// Context-chip utilities for the graph and inspection panel
// ---------------------------------------------------------------------------

/** Palette of visually distinct facet chip colors. */
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
