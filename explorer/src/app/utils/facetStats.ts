import type { DatasetVideo, Facet } from "@/app/types";

export type SelectedFacets = Record<string, { label: string; title: string }>;

export type FacetImpactStats = {
  relativeTotal: number; // observed unique contexts under filters
  theoreticalTotal: number; // product of vocab sizes for unfixed facets
  totalValues: number; // sum of vocab sizes across facets
};

export function facetToLabelKey(facet: Facet) {
  return facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
}

export function normalizeFacetVals(rawVals: string[] | undefined) {
  const vals = (rawVals || []).map((v) => (v ?? "").trim()).filter(Boolean);
  return vals.length ? vals : ["na"];
}

function pieceMatchesAllSelectedFacets(
  p: DatasetVideo["pieces"][number],
  selectedFacets: SelectedFacets
) {
  const entries = Object.entries(selectedFacets);
  if (entries.length === 0) return true;
  for (const [key, sel] of entries) {
    const vals = normalizeFacetVals((p.labels?.[key] || []) as string[]);
    if (!vals.includes(sel.label)) return false;
  }
  return true;
}

function pieceMatchesInfoTypeFilters(
  p: DatasetVideo["pieces"][number],
  selectedType: string,
  infoTypes: readonly string[]
) {
  // Always restrict to the enabled info types
  if (!infoTypes.includes(p.content_type as any)) return false;
  if (selectedType === "All") return true;
  return p.content_type === selectedType;
}

function buildContextKey(
  p: DatasetVideo["pieces"][number],
  facets: readonly Facet[]
) {
  const labels = (p.labels || {}) as Record<string, string[]>;
  if (!facets.length) return "__NO_FACETS__";

  const entries = facets
    .map((f) => facetToLabelKey(f))
    .filter(Boolean)
    .map((k) => {
      const rawVals = (labels[k] || []) as string[];
      const normalized = normalizeFacetVals(rawVals);
      return [k, normalized.sort().join("|")] as const;
    })
    .sort((a, b) => a[0].localeCompare(b[0]));

  return entries.map(([kk, v]) => `${kk}=${v}`).join(";");
}

/**
 * Discriminativeness D = Σ_c (n_c / N) · log₂(n_c).
 * D = 0   → perfect separation (each cell contains exactly one unit).
 * D = log₂(N) → no separation (all units in one cell).
 * Lower D is better. Adding more facets can only decrease or maintain D.
 */
export function computeD(
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

/**
 * High-level discriminativeness: groups pieces into cells by facet values,
 * collects distinct unit_ids per cell, and computes D.
 */
export function computeDiscriminativeness(
  pieces: DatasetVideo["pieces"],
  facets: readonly Facet[],
): { D: number; numUniqueUnits: number; cellCount: number } {
  const cellToUnits = new Map<string, Set<string>>();
  const allUnits = new Set<string>();

  for (const piece of pieces) {
    if (!piece.unit_id) continue;
    allUnits.add(piece.unit_id);
    const key = buildContextKey(piece, facets);
    let s = cellToUnits.get(key);
    if (!s) { s = new Set(); cellToUnits.set(key, s); }
    s.add(piece.unit_id);
  }

  const numUniqueUnits = allUnits.size;
  return {
    D: computeD(cellToUnits, numUniqueUnits),
    numUniqueUnits,
    cellCount: cellToUnits.size,
  };
}

/**
 * Computes context count stats under current selection filters:
 * - relativeTotal: observed unique contexts
 * - theoreticalTotal: product of vocab sizes for unfixed facets
 * - totalValues: sum of vocab sizes across facets
 *
 * For discriminativeness, use `computeDiscriminativeness` instead.
 */
export function computeFacetImpactStats({
  facets,
  selectedFacets,
  allPieces,
  selectedType,
  infoTypes,
}: {
  facets: readonly Facet[];
  selectedFacets: SelectedFacets;
  allPieces: DatasetVideo["pieces"];
  selectedType: string;
  infoTypes: readonly string[];
}): FacetImpactStats {
  const totalValues = facets.reduce((acc, f) => acc + (f.vocabulary?.length || 0), 0);

  const theoreticalTotal = facets.reduce((acc, f) => {
    const fKey = facetToLabelKey(f);
    const isFixed = Boolean(selectedFacets[fKey]);
    const len = f.vocabulary?.length || 0;
    return acc * (isFixed ? 1 : Math.max(1, len));
  }, 1);

  const contextTotals = new Set<string>();

  for (const p of allPieces) {
    if (!pieceMatchesAllSelectedFacets(p, selectedFacets)) continue;
    if (!pieceMatchesInfoTypeFilters(p, selectedType, infoTypes)) continue;
    contextTotals.add(buildContextKey(p, facets));
  }

  return {
    relativeTotal: contextTotals.size,
    theoreticalTotal,
    totalValues,
  };
}


