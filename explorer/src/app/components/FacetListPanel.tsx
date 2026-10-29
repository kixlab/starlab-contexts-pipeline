"use client";

import { useMemo, useState } from "react";
import type { Facet, DatasetVideoPiece } from "@/app/types";
import { computeDiscriminativeness } from "@/app/utils/facetStats";

// ── Exported sub-component: expandable facet detail content ──────────────────

export function FacetDetailContent({ facet }: { facet: Facet }) {
  return (
    <div className="mt-1 pl-7 pr-1 pb-2 space-y-2 text-[11px]">
      {/* Definition */}
      {facet.definition && (
        <div>
          <span className="text-neutral-500 font-medium">Definition: </span>
          <span className="text-neutral-700 dark:text-neutral-300">{facet.definition}</span>
        </div>
      )}

      {/* Guidelines */}
      {facet.guidelines && facet.guidelines.length > 0 && (
        <div>
          <span className="text-neutral-500 font-medium">Guidelines:</span>
          <ul className="mt-0.5 space-y-0.5 list-disc list-inside text-neutral-600 dark:text-neutral-400">
            {facet.guidelines.map((g, i) => (
              <li key={i} className="leading-snug">{g}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Vocabulary */}
      {facet.vocabulary.length > 0 && (
        <div>
          <span className="text-neutral-500 font-medium">Values ({facet.vocabulary.length}):</span>
          <div className="mt-1 space-y-1.5">
            {facet.vocabulary.map((v) => (
              <div key={v.label} className="border-l-2 border-neutral-300 dark:border-neutral-700 pl-2">
                <div className="flex items-baseline gap-1.5 flex-wrap">
                  <span className="text-neutral-800 dark:text-neutral-200 font-medium">{v.label}</span>
                  {v.definition && (
                    <span className="text-neutral-500">&mdash; {v.definition}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Exported sub-component: single facet row with checkbox + expand ──────────

export function FacetRow({
  facet,
  isSelected,
  onToggle,
  disabled = false,
  /** 0–1 intensity for background highlight based on |ΔH| relative to max */
  highlightIntensity = 0,
  deltaH,
}: {
  facet: Facet;
  isSelected: boolean;
  onToggle: () => void;
  disabled?: boolean;
  highlightIntensity?: number;
  deltaH?: number;
}) {
  const [expanded, setExpanded] = useState(false);

  // Compute background style for top-3 entropy highlight
  const highlightStyle = highlightIntensity > 0 && !isSelected
    ? { backgroundColor: `rgba(34, 197, 94, ${highlightIntensity})` }
    : undefined;

  return (
    <div
      className={`rounded transition-colors ${
        isSelected
          ? "bg-blue-50 dark:bg-blue-900/15 border border-blue-300 dark:border-blue-800/30"
          : "border border-transparent"
      } ${disabled && !isSelected ? "opacity-40" : ""}`}
      style={highlightStyle}
    >
      {/* Collapsed row */}
      <div
        className="flex items-center gap-1.5 px-2 py-1.5 cursor-pointer hover:bg-neutral-100 dark:hover:bg-white/5 rounded"
        onClick={() => setExpanded((v) => !v)}
      >
        {/* Checkbox */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (!disabled || isSelected) onToggle();
          }}
          className={`flex-shrink-0 w-4 h-4 rounded border flex items-center justify-center transition-colors ${
            disabled && !isSelected ? "cursor-not-allowed" : "cursor-pointer"
          }`}
          style={
            isSelected
              ? { borderColor: "#3B82F6", backgroundColor: "#2563EB" }
              : { borderColor: "#9CA3AF" }
          }
        >
          {isSelected && (
            <svg width={9} height={9} viewBox="0 0 10 10" fill="none" stroke="#fff" strokeWidth={2}>
              <path d="M1.5 5l3 3 4-5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>

        {/* Expand indicator */}
        <span className="text-neutral-400 dark:text-neutral-500 text-[10px] w-3 flex-shrink-0 select-none">
          {expanded ? "\u25BE" : "\u25B8"}
        </span>

        {/* Facet title */}
        <span
          className={`flex-1 text-xs truncate ${
            isSelected
              ? "text-blue-700 dark:text-blue-300 font-medium"
              : "text-neutral-700 dark:text-neutral-300"
          }`}
          title={facet.title}
        >
          {facet.title}
        </span>

        {/* ΔH (debug) */}
        {deltaH !== undefined && (
          <span className="text-[10px] text-neutral-400 dark:text-neutral-500 flex-shrink-0 tabular-nums">
            {deltaH < 0 ? "" : "+"}{deltaH.toFixed(3)}
          </span>
        )}

        {/* Value count */}
        <span className="text-[10px] text-neutral-400 dark:text-neutral-500 flex-shrink-0 tabular-nums">
          ({facet.vocabulary.length})
        </span>
      </div>

      {/* Expanded content */}
      {expanded && <FacetDetailContent facet={facet} />}
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

type FacetListPanelProps = {
  facets: Facet[];
  selectedFacetIds: Set<string>;
  onToggleFacet: (facetId: string) => void;
  /** Optional title shown above the list. Pass `null` to hide. */
  title?: string | null;
  className?: string;
  /** When set, disables selection of additional facets beyond this count. */
  maxSelectable?: number;
  /** Dataset pieces — when provided, enables per-facet ΔH computation & highlight. */
  pieces?: DatasetVideoPiece[];
};

export function FacetListPanel({
  facets,
  selectedFacetIds,
  onToggleFacet,
  title = "Facets",
  className = "",
  maxSelectable,
  pieces,
}: FacetListPanelProps) {
  const atLimit = maxSelectable !== undefined && selectedFacetIds.size >= maxSelectable;

  // Compute per-facet ΔH when pieces are available.
  // Baseline = currently selected facets. For each unselected facet, ΔH = what
  // adding it would do on top of the current selection.
  const selectedFacetsList = useMemo(
    () => facets.filter((f) => selectedFacetIds.has(f.id)),
    [facets, selectedFacetIds],
  );

  const facetDeltas = useMemo(() => {
    if (!pieces || pieces.length === 0) return new Map<string, number>();

    const baseD = computeDiscriminativeness(pieces, selectedFacetsList).D;

    const deltas = new Map<string, number>();
    for (const facet of facets) {
      if (selectedFacetIds.has(facet.id)) continue;
      const withD = computeDiscriminativeness(pieces, [...selectedFacetsList, facet]).D;
      // positive delta = reduces D = good; guaranteed >= 0
      const rawDelta = baseD - withD;
      const vocabSize = Math.max(1, facet.vocabulary.length);
      // deltas.set(facet.id, rawDelta / vocabSize);
      deltas.set(facet.id, rawDelta);
    }
    return deltas;
  }, [pieces, facets, selectedFacetsList, selectedFacetIds]);

  // Top-3 highlight ranks: rank 0 = strongest, rank 1 = second, rank 2 = third
  const facetRank = useMemo(() => {
    const ranks = new Map<string, number>();
    if (facetDeltas.size === 0) return ranks;
    const sorted = Array.from(facetDeltas.entries())
      .sort((a, b) => b[1] - a[1]); // highest positive first
    for (let i = 0; i < Math.min(3, sorted.length); i++) {
      ranks.set(sorted[i][0], i);
    }
    return ranks;
  }, [facetDeltas]);

  const sortedFacets = useMemo(
    () => [...facets].sort((a, b) => a.vocabulary.length - b.vocabulary.length),
    [facets],
  );

  if (facets.length === 0) {
    return <div className={`text-xs text-neutral-500 italic ${className}`}>No contexts available.</div>;
  }

  // Top-3 opacities: rank 0 → 30%, rank 1 → 20%, rank 2 → 10%
  const RANK_INTENSITY = [0.30, 0.20, 0.10] as const;

  return (
    <div className={className}>
      {title && (
        <div className="text-xs text-neutral-500 dark:text-neutral-400 font-medium mb-1.5 px-1">
          {title} ({selectedFacetIds.size}/{facets.length})
        </div>
      )}
      <div className="space-y-0.5">
        {sortedFacets.map((facet) => {
          const rank = facetRank.get(facet.id);
          const intensity = rank !== undefined ? RANK_INTENSITY[rank] : 0;
          return (
            <FacetRow
              key={facet.id}
              facet={facet}
              isSelected={selectedFacetIds.has(facet.id)}
              onToggle={() => onToggleFacet(facet.id)}
              disabled={atLimit && !selectedFacetIds.has(facet.id)}
              highlightIntensity={intensity}
              deltaH={facetDeltas.get(facet.id)}
            />
          );
        })}
      </div>
    </div>
  );
}
