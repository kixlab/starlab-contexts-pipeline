"use client";

import { useMemo, useState } from "react";
import { Tooltip } from "@/app/components/Tooltip";
import { FacetDetailContent } from "@/app/components/FacetListPanel";
import { FACET_VALUE_PERCENTAGE_EXPLANATION, DENOMINATOR_TOOLTIP, NO_CONTEXTS_MESSAGE, DENOMINATOR_TOOLTIP_RELATIVE, DENOMINATOR_TOOLTIP_THEORETICAL } from "@/app/constants/facetValueTooltip";
import type { Facet, DatasetVideo } from "@/app/types";
import { computeFacetImpactStats, computeDiscriminativeness, facetToLabelKey as facetToLabelKeyUtil, normalizeFacetVals as normalizeFacetValsUtil } from "@/app/utils/facetStats";

type SelectedFacets = Record<string, { label: string; title: string }>;

export function FacetPanel({
  facets,
  facetCandidates,
  onRemoveFacet,
  onAddFacet,
  allPieces,
  selectedFacets,
  setSelectedFacets,
  selectedType,
  infoTypes,
  denomMode,
  setDenomMode,
}: {
  facets: Facet[];
  facetCandidates: Facet[];
  onRemoveFacet: (facet: Facet) => void;
  onAddFacet: (facet: Facet) => void;
  allPieces: DatasetVideo["pieces"];
  selectedFacets: SelectedFacets;
  setSelectedFacets: (updater: (prev: SelectedFacets) => SelectedFacets) => void;
  selectedType: string;
  infoTypes: readonly string[];
  denomMode: 'theoretical' | 'relative';
  setDenomMode: (mode: 'theoretical' | 'relative') => void;
}) {
  const [hoverTip, setHoverTip] = useState<null | { x: number; y: number; title: string; text: string }>(null);
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [addQuery, setAddQuery] = useState("");
  const [expandedFacets, setExpandedFacets] = useState<Set<string>>(new Set());

  // New datasets use facet ids as label keys; old datasets used `context_${facet.title}`.
  const facetToLabelKey = (facet: Facet) => facetToLabelKeyUtil(facet);
  const normalizeFacetVals = (rawVals: string[] | undefined) => {
    const vals = (rawVals || []).map((v) => (v ?? "").trim()).filter(Boolean);
    return vals.length ? vals : ["na"];
  };

  const shownFacetIds = new Set(facets.map((f) => f.id || f.title));
  const addableCandidates = (facetCandidates || [])
    .filter((c) => !shownFacetIds.has(c.id || c.title))
    .filter((c) => {
      if (!addQuery.trim()) return true;
      const q = addQuery.trim().toLowerCase();
      return (
        (c.title || "").toLowerCase().includes(q) ||
        (c.id || "").toLowerCase().includes(q)
      );
    });

  const pieceMatchesFacetsExcept = (
    p: DatasetVideo["pieces"][number],
    excludeKey: string
  ) => {
    const entries = Object.entries(selectedFacets).filter(([k]) => k !== excludeKey);
    if (entries.length === 0) return true;
    for (const [key, sel] of entries) {
      const vals = normalizeFacetVals((p.labels?.[key] || []) as string[]);
      if (!vals.includes(sel.label)) return false;
    }
    return true;
  };

  // Filter pieces by selected facet values and info types (matching computeFacetImpactStats behavior)
  const filteredPieces = useMemo(() => {
    return allPieces.filter((p: DatasetVideo["pieces"][number]) => {
      if (!infoTypes.includes(p.content_type as any)) return false;
      for (const [key, sel] of Object.entries(selectedFacets)) {
        const vals = normalizeFacetValsUtil((p.labels?.[key] || []) as string[]);
        if (!vals.includes(sel.label)) return false;
      }
      return true;
    });
  }, [allPieces, infoTypes, selectedFacets]);

  // Discriminativeness D using the centralized formula
  const baseD = useMemo(
    () => computeDiscriminativeness(filteredPieces, facets).D,
    [filteredPieces, facets],
  );

  // Keep computeFacetImpactStats for relativeTotal/theoreticalTotal/totalValues display
  const baseImpact = computeFacetImpactStats({
    facets,
    selectedFacets,
    allPieces,
    selectedType: "All",
    infoTypes,
  });

  const formatDelta = (n: number) => {
    if (!Number.isFinite(n)) return "—";
    const sign = n > 0 ? "+" : "";
    return `${sign}${n}`;
  };

  const formatDeltaFloat = (n: number, digits = 2) => {
    if (!Number.isFinite(n)) return "—";
    const sign = n > 0 ? "+" : "";
    return `${sign}${n.toFixed(digits)}`;
  };

  const sortedCandidateImpacts = useMemo(() => {
    const MAX_CANDIDATES_FOR_SORT = 150;
    const candidates = addableCandidates.slice(0, MAX_CANDIDATES_FOR_SORT);

    const impacts = candidates.map((f) => {
      const withFacet = computeFacetImpactStats({
        facets: [...facets, f],
        selectedFacets,
        allPieces,
        selectedType: "All",
        infoTypes,
      });
      const withD = computeDiscriminativeness(filteredPieces, [...facets, f]).D;
      // Negative deltaD = good (D decreased)
      const deltaD = withD - baseD;
      return { facet: f, withFacet, deltaD };
    });

    impacts.sort((a, b) => {
      // Most negative ΔD (largest reduction) first
      if (a.deltaD !== b.deltaD) return a.deltaD - b.deltaD;
      return (a.facet.title || "").localeCompare(b.facet.title || "");
    });

    return { impacts, isCapped: addableCandidates.length > MAX_CANDIDATES_FOR_SORT, cap: MAX_CANDIDATES_FOR_SORT };
  }, [
    addableCandidates,
    filteredPieces,
    baseD,
    facets,
    allPieces,
    infoTypes,
    selectedFacets,
  ]);

  return (
    <section className="rounded-xl border border-neutral-200 dark:border-neutral-800 p-4 space-y-4 min-w-0 min-h-0 h-full flex flex-col">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs flex-wrap">
          {(() => {
            // Panel-level theoretical total under current selections
            const theoreticalTotalPanel = facets.reduce((acc, f) => {
              const fKey = facetToLabelKey(f);
              const isFixed = Boolean(selectedFacets[fKey]);
              const len = f.vocabulary?.length || 0;
              return acc * (isFixed ? 1 : Math.max(1, len));
            }, 1);
            // Panel-level relative total (observed) under current selections
            const contextsPanel = new Set<string>();
            for (const p of allPieces) {
              if (selectedType !== "All") {
                if (p.content_type !== selectedType) continue;
                if (!infoTypes.includes(p.content_type as any)) continue;
              }
              // must match all selected facets
              let ok = true;
              for (const [key, sel] of Object.entries(selectedFacets)) {
                const rawVals = (p.labels?.[key] || []) as string[];
                const vals = normalizeFacetVals(rawVals as string[]);
                if (!vals.includes(sel.label)) { ok = false; break; }
              }
              if (!ok) continue;
              const labels = (p.labels || {}) as Record<string, string[]>;
              const facetKeys =
                facets?.length
                  ? facets.map((f) => facetToLabelKey(f)).filter(Boolean)
                  : Object.keys(labels).filter((k) => k.startsWith("context_"));
              const entries = facetKeys
                .map((k) => {
                  const rawVals = (labels[k] || []) as string[];
                  const normalized = normalizeFacetVals(rawVals);
                  return [k, normalized.sort().join("|")] as const;
                })
                .sort((a, b) => a[0].localeCompare(b[0]));
              const kctx = entries.map(([kk, v]) => `${kk}=${v}`).join(";");
              if (kctx) contextsPanel.add(kctx);
            }
            const relativeTotalPanel = contextsPanel.size;
            return (
              <div className="flex items-center gap-2">
                <span className="text-neutral-500">stats:</span>

                <div className="inline-flex items-center gap-2">
                  <div className="inline-flex items-center gap-1 rounded-full border border-neutral-300 px-2 py-0.5 bg-white/70 dark:bg-neutral-950/40">
                    <span className="text-neutral-500">D(unit|ctx)</span>
                    <span className="font-medium text-neutral-800 dark:text-neutral-200">{baseD.toFixed(2)}</span>
                  </div>

                  <div className="inline-flex rounded-full border border-neutral-300 overflow-hidden">
                  <button
                    className={`px-2 py-0.5 ${denomMode === 'relative' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : ''}`}
                    onMouseEnter={(e) => {
                      setHoverTip({
                        x: e.clientX,
                        y: e.clientY + 16,
                        title: "Relative denominator",
                        text: DENOMINATOR_TOOLTIP_RELATIVE,
                      });
                    }}
                    onMouseMove={(e) => setHoverTip((t) => (t ? { ...t, x: e.clientX, y: e.clientY + 16 } : t))}
                    onMouseLeave={() => setHoverTip(null)}
                    onClick={() => setDenomMode('relative')}
                    aria-label="Use relative denominator"
                  >Relative ({relativeTotalPanel})</button>
                  <button
                    className={`px-2 py-0.5 ${denomMode === 'theoretical' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : ''}`}
                    onMouseEnter={(e) => {
                      setHoverTip({
                        x: e.clientX,
                        y: e.clientY + 16,
                        title: "Theoretical denominator",
                        text: DENOMINATOR_TOOLTIP_THEORETICAL,
                      });
                    }}
                    onMouseMove={(e) => setHoverTip((t) => (t ? { ...t, x: e.clientX, y: e.clientY + 16 } : t))}
                    onMouseLeave={() => setHoverTip(null)}
                    onClick={() => setDenomMode('theoretical')}
                    aria-label="Use theoretical denominator"
                  >Theoretical ({theoreticalTotalPanel})</button>
                  </div>

                  <div className="inline-flex items-center gap-1 rounded-full border border-neutral-300 px-2 py-0.5 bg-white/70 dark:bg-neutral-950/40">
                    <span className="text-neutral-500">values</span>
                    <span className="font-medium text-neutral-800 dark:text-neutral-200">{baseImpact.totalValues}</span>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
        <div className="flex items-center gap-2">
          <button
            className="flex items-center gap-1 text-xs p-1 rounded-full border border-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-900"
            onClick={() => setIsAddOpen(true)}
            aria-label="Add facet"
            title="Add facet"
          >
            <span className="px-1">+ facet</span>
          </button>
          {Object.keys(selectedFacets).length > 0 && (
            <button
              className="flex items-center gap-1 text-xs p-1 rounded-full border border-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-900"
              onClick={() => setSelectedFacets(() => ({} as SelectedFacets))}
              aria-label="clean all"
              title="clean all"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path d="M9 3a1 1 0 0 0-1 1v1H5.5a1 1 0 1 0 0 2H6v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V7h.5a1 1 0 1 0 0-2H16V4a1 1 0 0 0-1-1H9zm2 3h2V4h-2v2zM8 7h10v12a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V7zm3 3a1 1 0 1 0-2 0v8a1 1 0 1 0 2 0v-8zm5 0a1 1 0 1 0-2 0v8a1 1 0 1 0 2 0v-8z"/>
              </svg>
              <span className="pr-1">clean all</span>
            </button>
          )}
        </div>
      </div>
      <div className="space-y-6 flex-1 overflow-y-auto">
        {facets.length === 0 && (
          <div className="text-sm text-neutral-500 border border-dashed border-neutral-300 dark:border-neutral-700 rounded-lg p-4">
            No facets selected yet. Click <span className="font-medium text-neutral-700 dark:text-neutral-200">+ facet</span> to add from candidates.
          </div>
        )}
        {facets.map((facet) => (
          <div key={facet.id || facet.title} className="space-y-0">
            <div className="grid grid-cols-[90px_1fr] gap-3 items-start">
            <div
              className="text-sm font-medium leading-5 pt-1 flex items-start justify-between gap-2 cursor-help"
              onMouseEnter={(e) =>
                setHoverTip({
                  x: e.clientX,
                  y: e.clientY + 16,
                  title: facet.title,
                  text: String(facet.definition ?? ""),
                })
              }
              onMouseMove={(e) => setHoverTip((t) => (t ? { ...t, x: e.clientX, y: e.clientY + 16 } : t))}
              onMouseLeave={() => setHoverTip(null)}
            >
              <div className="flex items-center gap-1">
                <button
                  className="text-neutral-400 dark:text-neutral-500 text-[10px] flex-shrink-0 select-none hover:text-neutral-600 dark:hover:text-neutral-300"
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedFacets((prev) => {
                      const next = new Set(prev);
                      const key = facet.id || facet.title;
                      if (next.has(key)) next.delete(key); else next.add(key);
                      return next;
                    });
                  }}
                  title="Show definition, guidelines, and values"
                >
                  {expandedFacets.has(facet.id || facet.title) ? "\u25BE" : "\u25B8"}
                </button>
                <span className="break-words max-w-[68px]">
                  {facet.title}
                </span>
              </div>
              <div className="flex items-center gap-1">
                {(() => {
                  const key = facetToLabelKey(facet);
                  const current = selectedFacets[key]?.label;
                  return current ? (
                    <button
                      className="text-[10px] p-1 rounded-full border border-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                      onClick={() =>
                        setSelectedFacets((prev) => {
                          const next = { ...prev } as SelectedFacets;
                          delete next[key];
                          return next;
                        })
                      }
                      aria-label={`Clear ${facet.title}`}
                      title="Clear selection"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5">
                        <path d="M9 3a1 1 0 0 0-1 1v1H5.5a1 1 0 1 0 0 2H6v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V7h.5a1 1 0 1 0 0-2H16V4a1 1 0 0 0-1-1H9zm2 3h2V4h-2v2zM8 7h10v12a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V7zm3 3a1 1 0 1 0-2 0v8a1 1 0 1 0 2 0v-8zm5 0a1 1 0 1 0-2 0v8a1 1 0 1 0 2 0v-8z"/>
                      </svg>
                    </button>
                  ) : null;
                })()}
                <button
                  className="text-[10px] p-1 rounded-full border border-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                  onClick={(e) => { e.stopPropagation(); onRemoveFacet(facet); }}
                  aria-label={`Remove facet ${facet.title}`}
                  title="Remove facet"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5">
                    <path d="M6 7a1 1 0 0 1 1-1h10a1 1 0 1 1 0 2H7a1 1 0 0 1-1-1z"/>
                  </svg>
                </button>
              </div>
            </div>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {facet.vocabulary.map((a) => (
                <div
                  key={a.label}
                  onClick={() => {
                    const key = facetToLabelKey(facet);
                    setSelectedFacets((prev) => {
                      const current = prev[key]?.label;
                      const next = { ...prev } as SelectedFacets;
                      if (current === a.label) {
                        delete next[key];
                      } else {
                        next[key] = { label: a.label, title: facet.title };
                      }
                      return next;
                    });
                  }}
                  onMouseEnter={(e) => {
                    // Calculate percentage for tooltip
                    const key = facetToLabelKey(facet);
                    const rowHasSelection = Boolean(selectedFacets[key]);
                    let tooltipText = a.definition;
                    
                    if (!rowHasSelection) {
                      // Filter pieces to those matching all currently selected facets except this row
                      const matchesOtherSelections = allPieces.filter((p) => pieceMatchesFacetsExcept(p, key));
                      
                      const buildContextKey = (p: DatasetVideo["pieces"][number]) => {
                        const labels = (p.labels || {}) as Record<string, string[]>;
                        const facetKeys =
                          facets?.length
                            ? facets.map((f) => facetToLabelKey(f)).filter(Boolean)
                            : Object.keys(labels).filter((k) => k.startsWith("context_"));
                        const entries = facetKeys
                          .map((k) => {
                            const rawVals = (labels[k] || []) as string[];
                            const normalized = normalizeFacetVals(rawVals);
                            return [k, normalized.sort().join("|")] as const;
                          })
                          .sort((a, b) => a[0].localeCompare(b[0]));
                        return entries.map(([kk, v]) => `${kk}=${v}`).join(";");
                      };

                      // Contexts for this value, restricted by other selected facets
                      const contextsForValue = new Map<string, Set<string>>();
                      for (const p of matchesOtherSelections) {
                        if (selectedType !== "All") {
                          if (p.content_type !== selectedType) continue;
                          if (!infoTypes.includes(p.content_type as any)) continue;
                        }
                        const raw = (p.labels?.[key] || []) as string[];
                        const vals = normalizeFacetVals(raw);
                        const hasCandidate = vals.includes(a.label);
                        if (!hasCandidate) continue;
                        const kctx = buildContextKey(p);
                        if (!kctx) continue;
                        const set = contextsForValue.get(kctx) || new Set<string>();
                        if (infoTypes.includes(p.content_type as any)) set.add(p.content_type);
                        contextsForValue.set(kctx, set);
                      }

                      // Theoretical total contexts for this value
                      const theoreticalTotal = facets
                        .filter((f) => f.title !== facet.title)
                        .reduce((acc, f) => {
                          const fKey = facetToLabelKey(f);
                          const isFixed = Boolean(selectedFacets[fKey]);
                          const len = f.vocabulary?.length || 0;
                          return acc * (isFixed ? 1 : Math.max(1, len));
                        }, 1);

                      const relativeTotal = contextsForValue.size;
                      const denom = denomMode === 'relative' ? relativeTotal : theoreticalTotal;
                      
                      if (denom > 0) {
                        // Calculate coverage based on selected information type
                        let pct: number;
                        if (selectedType === "All") {
                          // Average coverage across all types
                          let sumFractions = 0;
                          for (const t of infoTypes) {
                            let countWithType = 0;
                            for (const set of contextsForValue.values()) if (set.has(t)) countWithType += 1;
                            sumFractions += countWithType / denom;
                          }
                          const avg = sumFractions / (infoTypes.length || 1);
                          pct = Math.round(avg * 100);
                        } else {
                          // Coverage for specific type
                          let countWithType = 0;
                          for (const set of contextsForValue.values()) if (set.has(selectedType)) countWithType += 1;
                          pct = Math.round((countWithType / denom) * 100);
                        }
                        
                        // Replace placeholders in explanation
                        const explanation = FACET_VALUE_PERCENTAGE_EXPLANATION
                          .replace(/{infoCoverage}/g, pct.toString());
                        tooltipText = [tooltipText, "", explanation].filter(Boolean).join("\n\n");
                      } else {
                        tooltipText = [tooltipText, "", NO_CONTEXTS_MESSAGE].filter(Boolean).join("\n\n");
                      }
                    }
                    
                    setHoverTip({
                      x: e.clientX,
                      y: e.clientY + 16,
                      title: a.label,
                      text: tooltipText,
                    });
                  }}
                  onMouseMove={(e) => setHoverTip((t) => (t ? { ...t, x: e.clientX, y: e.clientY + 16 } : t))}
                  onMouseLeave={() => {
                    setHoverTip(null);
                  }}
                  className={`group relative shrink-0 px-3 py-2 rounded-xl border cursor-pointer select-none ${
                    (() => {
                      const key = facetToLabelKey(facet);
                      const sel = selectedFacets[key]?.label;
                      return sel === a.label;
                    })()
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                      : (() => {
                          const key = facetToLabelKey(facet);
                          const hasSel = Boolean(selectedFacets[key]);
                          return hasSel
                            ? "border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-900/60 opacity-50"
                            : "border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 hover:shadow";
                        })()
                  }`}
                >
                  <div className="flex flex-col min-w-[7.5rem] max-w-[10rem]">
                    <span className="whitespace-nowrap text-sm font-medium truncate">{a.label}</span>
                    {(() => {
                      const key = facetToLabelKey(facet);
                      const rowHasSelection = Boolean(selectedFacets[key]);
                      if (rowHasSelection) return <span className="mt-1 h-4" />;
                      // Build contexts under current selections (excluding this row)
                      const matchesOtherSelections = allPieces.filter((p) => pieceMatchesFacetsExcept(p, key));

                      const buildContextKey = (p: DatasetVideo["pieces"][number]) => {
                        const labels = (p.labels || {}) as Record<string, string[]>;
                        const facetKeys =
                          facets?.length
                            ? facets.map((f) => facetToLabelKey(f)).filter(Boolean)
                            : Object.keys(labels).filter((k) => k.startsWith("context_"));
                        const entries = facetKeys
                          .map((k) => {
                            const rawVals = (labels[k] || []) as string[];
                            const normalized = normalizeFacetVals(rawVals);
                            return [k, normalized.sort().join("|")] as const;
                          })
                          .sort((a, b) => a[0].localeCompare(b[0]));
                        return entries.map(([kk, v]) => `${kk}=${v}`).join(";");
                      };

                      // Contexts for this value, restricted by other selected facets
                      const contextsForValue = new Map<string, Set<string>>();
                      for (const p of matchesOtherSelections) {
                        if (selectedType !== "All") {
                          if (p.content_type !== selectedType) continue;
                          if (!infoTypes.includes(p.content_type as any)) continue;
                        }
                        const raw = (p.labels?.[key] || []) as string[];
                        const vals = normalizeFacetVals(raw);
                        const hasCandidate = vals.includes(a.label);
                        if (!hasCandidate) continue;
                        const kctx = buildContextKey(p);
                        if (!kctx) continue;
                        const set = contextsForValue.get(kctx) || new Set<string>();
                        // only count allowed info types
                        if (infoTypes.includes(p.content_type as any)) set.add(p.content_type);
                        contextsForValue.set(kctx, set);
                      }

                      // Theoretical total contexts for this value: product across other facets that are NOT selected (selected ones are fixed)
                      const theoreticalTotal = facets
                        .filter((f) => f.title !== facet.title)
                        .reduce((acc, f) => {
                          const fKey = facetToLabelKey(f);
                          const isFixed = Boolean(selectedFacets[fKey]);
                          const len = f.vocabulary?.length || 0;
                          return acc * (isFixed ? 1 : Math.max(1, len));
                        }, 1);

                      const relativeTotal = contextsForValue.size;
                      const denom = (typeof denomMode !== 'undefined' && denomMode === 'relative') ? relativeTotal : theoreticalTotal;
                      if (denom === 0) {
                        return (
                          <div className="flex items-center gap-2 mt-1 h-4">
                            <div className="relative w-4 h-4">
                              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="9" className="text-neutral-300 dark:text-neutral-700" />
                                <line x1="6" y1="18" x2="18" y2="6" className="text-neutral-500 dark:text-neutral-400" />
                              </svg>
                            </div>
                            <span className="text-[10px] text-neutral-500 dark:text-neutral-400">no contexts</span>
                          </div>
                        );
                      }

                      // Calculate coverage based on selected information type
                      let pct: number;
                      if (selectedType === "All") {
                        // Average coverage across all types
                        let sumFractions = 0;
                        for (const t of infoTypes) {
                          let countWithType = 0;
                          for (const set of contextsForValue.values()) if (set.has(t)) countWithType += 1;
                          sumFractions += countWithType / denom;
                        }
                        const avg = sumFractions / (infoTypes.length || 1);
                        pct = Math.round(avg * 100);
                      } else {
                        // Coverage for specific type
                        let countWithType = 0;
                        for (const set of contextsForValue.values()) if (set.has(selectedType)) countWithType += 1;
                        pct = Math.round((countWithType / denom) * 100);
                      }
                      return (
                        <div className="flex items-center gap-2 mt-1 h-4">
                          <div className="relative w-4 h-4">
                            <svg className="w-4 h-4 transform -rotate-90" viewBox="0 0 36 36">
                              <path
                                className="text-neutral-200 dark:text-neutral-700"
                                stroke="currentColor"
                                strokeWidth="3"
                                fill="none"
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              />
                              <path
                                className="text-blue-500"
                                stroke="currentColor"
                                strokeWidth="3"
                                fill="none"
                                strokeDasharray={`${pct}, 100`}
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                              />
                            </svg>
                          </div>
                          <span className="text-[10px] text-neutral-600 dark:text-neutral-400 font-medium">
                            {pct}%
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                </div>
              ))}
            </div>
          </div>
            {expandedFacets.has(facet.id || facet.title) && (
              <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/40 mx-1 mb-1">
                <FacetDetailContent facet={facet} />
              </div>
            )}
          </div>
        ))}
      </div>
      {isAddOpen && (
        <div className="fixed inset-0 z-[1100] bg-black/30 flex items-center justify-center p-4" onMouseDown={() => setIsAddOpen(false)}>
          <div
            className="w-full max-w-lg rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 p-4 shadow-2xl"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold">Add a facet</div>
                <div className="text-xs text-neutral-500">Choose from `facet_candidates`</div>
              </div>
              <button
                className="text-xs rounded-full border border-neutral-300 px-2 py-1 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                onClick={() => setIsAddOpen(false)}
              >
                Close
              </button>
            </div>

            <div className="mt-3">
              <input
                value={addQuery}
                onChange={(e) => setAddQuery(e.target.value)}
                placeholder="Search facets…"
                className="w-full text-sm rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2"
              />
            </div>

            <div className="mt-3 rounded-lg border border-neutral-200 dark:border-neutral-800 p-3 bg-neutral-50 dark:bg-neutral-900/40">
              <div className="text-xs font-medium text-neutral-800 dark:text-neutral-200">Current context stats</div>
              <div className="mt-1 grid grid-cols-3 gap-2 text-[11px] text-neutral-600 dark:text-neutral-400">
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-neutral-500">D(unit|ctx)</div>
                  <div className="font-medium text-neutral-800 dark:text-neutral-200">{baseD.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-neutral-500">Relative ctx</div>
                  <div className="font-medium text-neutral-800 dark:text-neutral-200">{baseImpact.relativeTotal}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-neutral-500">Theoretical ctx</div>
                  <div className="font-medium text-neutral-800 dark:text-neutral-200">{baseImpact.theoreticalTotal}</div>
                </div>
              </div>
            </div>

            <div className="mt-3 max-h-[50vh] overflow-y-auto space-y-2">
              {addableCandidates.length === 0 ? (
                <div className="text-sm text-neutral-500 py-6 text-center">No facets to add.</div>
              ) : (
                sortedCandidateImpacts.impacts.slice(0, 50).map(({ facet: f, withFacet, deltaD }, idx) => {
                  const deltaRelative = withFacet.relativeTotal - baseImpact.relativeTotal;
                  const deltaTheoretical = withFacet.theoreticalTotal - baseImpact.theoreticalTotal;
                  const deltaValues = withFacet.totalValues - baseImpact.totalValues;

                  return (
                    <button
                      key={f.id || f.title}
                      className="w-full text-left rounded-lg border border-neutral-200 dark:border-neutral-800 p-3 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                      onClick={() => {
                        onAddFacet(f);
                        setIsAddOpen(false);
                        setAddQuery("");
                      }}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-sm font-medium truncate">
                            <span className="mr-2 inline-flex w-7 justify-end text-neutral-400 tabular-nums">
                              {idx + 1}.
                            </span>
                            {f.title}
                          </div>
                          {/* <div className="text-xs text-neutral-500 truncate">{f.id || "no id"}</div> */}
                           {String(f.definition ?? "").trim() && (
                             <div className="mt-1 text-[11px] text-neutral-600 dark:text-neutral-400 line-clamp-3 whitespace-pre-wrap">
                               {String(f.definition ?? "")}
                             </div>
                           )}
                        </div>
                        <div className="shrink-0 text-right">
                          <div className="text-[10px] text-neutral-500">Δ D</div>
                          <div className="text-xs font-semibold text-neutral-800 dark:text-neutral-200">
                            {formatDeltaFloat(deltaD, 2)}
                          </div>
                        </div>
                      </div>

                      <div className="mt-2 grid grid-cols-3 gap-2 text-[11px] text-neutral-600 dark:text-neutral-400">
                        <div className="rounded-md border border-neutral-200 dark:border-neutral-800 px-2 py-1 bg-white/60 dark:bg-neutral-950/40">
                          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Relative</div>
                          <div className="font-medium text-neutral-800 dark:text-neutral-200">
                            {withFacet.relativeTotal} ({formatDelta(deltaRelative)})
                          </div>
                        </div>
                        <div className="rounded-md border border-neutral-200 dark:border-neutral-800 px-2 py-1 bg-white/60 dark:bg-neutral-950/40">
                          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Theoretical</div>
                          <div className="font-medium text-neutral-800 dark:text-neutral-200">
                            {withFacet.theoreticalTotal} ({formatDelta(deltaTheoretical)})
                          </div>
                        </div>
                        <div className="rounded-md border border-neutral-200 dark:border-neutral-800 px-2 py-1 bg-white/60 dark:bg-neutral-950/40">
                          <div className="text-[10px] uppercase tracking-wide text-neutral-500">Values</div>
                          <div className="font-medium text-neutral-800 dark:text-neutral-200">
                            {withFacet.totalValues} ({formatDelta(deltaValues)})
                          </div>
                        </div>
                      </div>
                    </button>
                  );
                })
              )}
              {(sortedCandidateImpacts.impacts.length > 50 || sortedCandidateImpacts.isCapped) && (
                <div className="text-[11px] text-neutral-500 py-2 text-center">
                  {sortedCandidateImpacts.isCapped
                    ? `Sorted by ΔD using the first ${sortedCandidateImpacts.cap} matches. Narrow your search to sort more precisely.`
                    : "Showing the top 50 (sorted by ΔD). Narrow your search to see others."}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      {hoverTip && <Tooltip tip={hoverTip} />}
    </section>
  );
}

