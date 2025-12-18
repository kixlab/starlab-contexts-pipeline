import type { DatasetVideo, Facet } from "@/app/types";

export type SelectedFacets = Record<string, { label: string; title: string }>;

export type FacetImpactStats = {
  relativeTotal: number; // observed unique contexts under filters
  theoreticalTotal: number; // product of vocab sizes for unfixed facets
  totalValues: number; // sum of vocab sizes across facets
  conditionalEntropyBits: number; // H(Information | context) in bits
};

export type DiscriminativenessTarget = "unit" | "piece" | "type";

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

function log2(x: number) {
  return Math.log(x) / Math.log(2);
}

function entropyBitsFromCounts(counts: readonly number[]) {
  let total = 0;
  for (const c of counts) total += c;
  if (total <= 0) return 0;
  let h = 0;
  for (const c of counts) {
    if (c <= 0) continue;
    const p = c / total;
    h += -p * log2(p);
  }
  return h;
}

/**
 * Computes:
 * - Relative/theoretical context totals under current selection filters
 * - Total values across facets (sum of vocab sizes)
 * - Discriminativeness as conditional entropy H(target | context)
 *
 * Definitions:
 * - context: joint assignment of values across ALL provided facets (each piece contributes one context key).
 * - target:
 *   - "type": the `content_type` label (restricted to enabled `infoTypes`).
 *   - "piece": the `piece_id` label (i.e., which exact information piece it is).
 *   - "unit": the `unit_id` label (i.e., which information unit it is).
 * - Conditioning distribution p(context): treated as uniform over observed contexts (each context equally likely).
 */
export function computeFacetImpactStats({
  facets,
  selectedFacets,
  allPieces,
  selectedType,
  infoTypes,
  target = "type",
}: {
  facets: readonly Facet[];
  selectedFacets: SelectedFacets;
  allPieces: DatasetVideo["pieces"];
  selectedType: string;
  infoTypes: readonly string[];
  target?: DiscriminativenessTarget;
}): FacetImpactStats {
  const totalValues = facets.reduce((acc, f) => acc + (f.vocabulary?.length || 0), 0);

  const theoreticalTotal = facets.reduce((acc, f) => {
    const fKey = facetToLabelKey(f);
    const isFixed = Boolean(selectedFacets[fKey]);
    const len = f.vocabulary?.length || 0;
    return acc * (isFixed ? 1 : Math.max(1, len));
  }, 1);

  // contextKey -> totalPiecesInContext, and counts by target label
  const contextTotals = new Map<string, number>();
  const contextTargetCounts = new Map<string, Map<string, number>>();

  for (const p of allPieces) {
    if (!pieceMatchesAllSelectedFacets(p, selectedFacets)) continue;
    if (!pieceMatchesInfoTypeFilters(p, selectedType, infoTypes)) continue;

    const ctx = buildContextKey(p, facets);
    contextTotals.set(ctx, (contextTotals.get(ctx) || 0) + 1);

    const label =
      target === "unit"
        ? p.unit_id
        : target === "piece"
          ? p.piece_id
          : p.content_type;
    const byLabel = contextTargetCounts.get(ctx) || new Map<string, number>();
    byLabel.set(label, (byLabel.get(label) || 0) + 1);
    contextTargetCounts.set(ctx, byLabel);
  }

  const relativeTotal = contextTotals.size;

  // H(target | Context) = (1/|C|) * sum_c H(target | c)   (uniform over contexts)
  if (relativeTotal <= 0) {
    return { relativeTotal, theoreticalTotal, totalValues, conditionalEntropyBits: 0 };
  }

  let conditionalEntropyBits = 0;
  for (const [ctx] of contextTotals.entries()) {
    const byLabel = contextTargetCounts.get(ctx) || new Map<string, number>();
    const counts =
      target === "piece" || target === "unit"
        ? Array.from(byLabel.values())
        : infoTypes.map((t) => byLabel.get(t) || 0);
    const hCtx = entropyBitsFromCounts(counts);
    conditionalEntropyBits += hCtx;
  }
  conditionalEntropyBits = conditionalEntropyBits / relativeTotal;

  return {
    relativeTotal,
    theoreticalTotal,
    totalValues,
    conditionalEntropyBits,
  };
}


