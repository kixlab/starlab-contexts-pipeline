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
 * Discriminativeness D = H(U|C) — item-level conditional entropy.
 *
 *   H(U|C) = Σ_c P(C=c) · H(U|C=c)
 *   where:
 *     P(C=c)      = m_c / m          (fraction of items in cell c)
 *     P(U=u|C=c)  = count_{u,c} / m_c
 *     H(U|C=c)    = -Σ_u P(U=u|C=c) · log₂ P(U=u|C=c)
 *
 * D = 0      → perfect separation (each cell has items from only one unit).
 * D = H(U)   → no separation (context provides no info about unit).
 * Lower D is better.
 *
 * @param cellToUnits  cell signature → { unit_id → item_count }
 * @param numUnits     unused, kept for backwards compatibility
 */
export function computeD(
  cellToUnits: Map<string, Map<string, number>>,
  numUnits: number = 0
): number {
  let m = 0;
  for (const unitCounts of cellToUnits.values()) {
    for (const count of unitCounts.values()) {
      m += count;
    }
  }
  if (m === 0) return 0;

  let total = 0;
  for (const unitCounts of cellToUnits.values()) {
    let m_c = 0;
    for (const count of unitCounts.values()) {
      m_c += count;
    }
    if (m_c === 0) continue;
    const p_c = m_c / m;
    let h_c = 0;
    for (const count of unitCounts.values()) {
      if (count > 0) {
        const p_u_given_c = count / m_c;
        h_c -= p_u_given_c * Math.log2(p_u_given_c);
      }
    }
    total += p_c * h_c;
  }
  return total;
}

/**
 * Compute marginal unit entropy H(U) = -Σ_u P(U=u)·log₂ P(U=u).
 * This is the data-dependent upper bound for H(U|C).
 */
export function computeH_U(pieces: DatasetVideo["pieces"]): number {
  const unitCounts = new Map<string, number>();
  let m = 0;
  for (const piece of pieces) {
    if (!piece.unit_id) continue;
    unitCounts.set(piece.unit_id, (unitCounts.get(piece.unit_id) ?? 0) + 1);
    m++;
  }
  if (m === 0) return 0;
  let h = 0;
  for (const count of unitCounts.values()) {
    const p = count / m;
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/**
 * High-level discriminativeness: groups pieces into cells by facet values,
 * counts items per unit per cell, and computes D = H(U|C).
 *
 * Also returns H_U (marginal unit entropy) — the data-dependent upper bound
 * for D. Normalized discriminativeness = D / H_U ∈ [0, 1].
 */
export function computeDiscriminativeness(
  pieces: DatasetVideo["pieces"],
  facets: readonly Facet[],
): { D: number; H_U: number; numUniqueUnits: number; cellCount: number } {
  const cellToUnits = new Map<string, Map<string, number>>();
  const allUnits = new Map<string, number>();

  for (const piece of pieces) {
    if (!piece.unit_id) continue;
    allUnits.set(piece.unit_id, (allUnits.get(piece.unit_id) ?? 0) + 1);
    const key = buildContextKey(piece, facets);
    let unitCounts = cellToUnits.get(key);
    if (!unitCounts) { unitCounts = new Map(); cellToUnits.set(key, unitCounts); }
    unitCounts.set(piece.unit_id, (unitCounts.get(piece.unit_id) ?? 0) + 1);
  }

  const numUniqueUnits = allUnits.size;
  const m = Array.from(allUnits.values()).reduce((a, b) => a + b, 0);
  let H_U = 0;
  if (m > 0) {
    for (const count of allUnits.values()) {
      const p = count / m;
      if (p > 0) H_U -= p * Math.log2(p);
    }
  }

  return {
    D: computeD(cellToUnits, numUniqueUnits),
    H_U,
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


