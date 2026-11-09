import type { Facet } from "@/app/types";

export function facetToLabelKey(facet: Facet) {
  return facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
}

export function normalizeFacetVals(rawVals: string[] | undefined) {
  const vals = (rawVals || []).map((v) => (v ?? "").trim()).filter(Boolean);
  return vals.length ? vals : ["na"];
}
