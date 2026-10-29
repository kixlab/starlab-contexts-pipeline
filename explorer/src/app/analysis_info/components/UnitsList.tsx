import { useEffect, useMemo, useRef, useState } from "react";
import type { DatasetVideo, Facet } from "@/app/types";
import { ThumbnailMentions } from "./ThumbnailMentions";
import { memo } from "react";

export function UnitsList({
  dataset,
  allPieces,
  facets,
  selectedType,
  pieceMatchesFacet,
  selectedUnitId,
  onSelect,
  infoTypes,
  selectedFacets,
}: {
  dataset: DatasetVideo[];
  allPieces: DatasetVideo["pieces"];
  facets: Facet[];
  selectedType: string;
  pieceMatchesFacet: (p: DatasetVideo["pieces"][number]) => boolean;
  selectedUnitId: string | null;
  onSelect: (unitId: string) => void;
  infoTypes: readonly string[];
  selectedFacets: Record<string, { label: string; title: string }>;
}) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const selectedUnitRef = useRef<HTMLDivElement>(null);
  const [stickyPosition, setStickyPosition] = useState<'top' | 'bottom' | 'none' | null>(null);
  const [selectedUnitData, setSelectedUnitData] = useState<{unitId: string, info: any} | null>(null);

  // Build a global catalog of facets and their possible values (stable order matching FacetPanel)
  const facetCatalog = useMemo(() => {
    const facetKeyToTitle = new Map<string, string>();
    const facetKeyToValues = new Map<string, Set<string>>();
    const facetToLabelKey = (facet: Facet) =>
      facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
    const facetKeysInOrder = facets.length ? facets.map(facetToLabelKey) : [];
    const facetKeyToFacet = new Map<string, Facet>(facets.map((f) => [facetToLabelKey(f), f]));

    const addValue = (key: string, title: string, value: string) => {
      facetKeyToTitle.set(key, title);
      const set = facetKeyToValues.get(key) || new Set<string>();
      set.add(value);
      facetKeyToValues.set(key, set);
    };

    // Scan all pieces to build catalog (independent of current filters)
    (allPieces || []).forEach((p) => {
      const labels = (p.labels || {}) as Record<string, string[]>;
      const keys =
        facetKeysInOrder.length > 0
          ? facetKeysInOrder
          : Object.keys(labels).filter((k) => k.startsWith("context_"));
      keys
        .forEach((k) => {
          const rawVals = (labels[k] || []) as string[];
          const vals = rawVals.map((v) => (v ?? "").trim());
          const title = facetKeyToFacet.get(k)?.title ?? k.replace("context_", "");
          const clean = vals.filter(Boolean);
          if (clean.length === 0) addValue(k, title, "na");
          else clean.forEach((v) => addValue(k, title, v));
        });
    });

    // Convert to arrays and order facets according to the provided facets list
    const order = facets.map(facetToLabelKey);
    const catalog = Array.from(facetKeyToValues.entries())
      .sort((a, b) => {
        const ia = order.indexOf(a[0]);
        const ib = order.indexOf(b[0]);
        if (ia === -1 && ib === -1) return a[0].localeCompare(b[0]);
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
      })
      .map(([key, set]) => {
        const title = facetKeyToTitle.get(key) || key.replace("context_", "");
        // Use facet labels order if available; otherwise fallback to alphabetical with `na` last
        const facetDef = facetKeyToFacet.get(key);
        let values = Array.from(set.values());
        if (facetDef?.vocabulary?.length) {
          const labelOrder = [...facetDef.vocabulary.map((a) => a.label)];
          values.sort((a, b) => {
            const ia = labelOrder.indexOf(a);
            const ib = labelOrder.indexOf(b);
            if (ia === -1 && ib === -1) return a.localeCompare(b);
            if (ia === -1) return 1;
            if (ib === -1) return -1;
            return ia - ib;
          });
        } else {
          values = values.sort((a, b) => {
            const aNA = a === "na" ? 1 : 0;
            const bNA = b === "na" ? 1 : 0;
            if (aNA !== bNA) return aNA - bNA; // place `na` last
            return a.localeCompare(b);
          });
        }
        return { key, title, values } as const;
      });

    return catalog;
  }, [allPieces, facets]);

  // Compact facet row component
  const FacetValuesCompact = memo(function FacetValuesCompact({
    unitId,
    facetCatalog,
    facetToValues,
    selectedFacets,
    collapsible = false,
    defaultCollapsed = false,
  }: {
    unitId: string;
    facetCatalog: ReadonlyArray<{ key: string; title: string; values: string[] }>;
    facetToValues: Map<string, Set<string>>;
    selectedFacets: Record<string, { label: string; title: string }>;
    collapsible?: boolean;
    defaultCollapsed?: boolean;
  }) {
    const [expanded, setExpanded] = useState<boolean>(!collapsible || !defaultCollapsed);
    const rows = useMemo(() => {
      return facetCatalog.map(({ key, title, values }) => {
        const unitValues = facetToValues.get(key) || new Set<string>();
        const selected = selectedFacets[key]?.label;
        const displayValues = values.filter((v) => unitValues.has(v));
        const content = displayValues.length ? displayValues : [];
        return {
          key,
          title,
          values: content,
          selected,
          isRelevant: Boolean(selected),
        };
      });
    }, [facetCatalog, facetToValues, selectedFacets]);

    return (
      <div className="mt-2 text-[10px] text-neutral-600 dark:text-neutral-300 w-full">
        <div 
          className={`font-medium mb-1 ${collapsible ? 'cursor-pointer select-none hover:underline' : ''}`}
          onClick={(e) => { if (collapsible) { e.stopPropagation(); setExpanded((v)=>!v); } }}
        >
          Task contexts{collapsible ? (expanded ? ' (hide)' : ' (show)') : ''}:
        </div>
        {expanded && (
        <div className="space-y-0.5">
          {rows.map(({ key, title, values, selected }) => (
            <div key={`${unitId}-${key}`} className="grid grid-cols-[auto_1fr] gap-2 w-full items-start">
              <div className="text-neutral-500 whitespace-nowrap">{title}:</div>
              <div className="min-w-0 flex flex-wrap gap-x-1">
                {values.map((val, idx) => {
                  const isSelected = selected === val;
                  return (
                    <span key={`${unitId}-${key}-${val}`} className="flex items-center gap-1">
                      <span
                        className={
                          isSelected
                            ? 'px-1 py-0.5 rounded border border-blue-300 dark:border-blue-700 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 font-semibold'
                            : 'text-neutral-700 dark:text-neutral-200'
                        }
                      >
                        {val}
                      </span>
                      {idx < values.length - 1 && (
                        <span className="text-neutral-400 dark:text-neutral-500" aria-hidden>/</span>
                      )}
                    </span>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        )}
      </div>
    );
  });

  // Track selected unit position relative to visible area
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !selectedUnitId || !selectedUnitRef.current) return;

    const handleScroll = () => {
      const selectedElement = selectedUnitRef.current;
      if (!selectedElement) return;

      const containerRect = container.getBoundingClientRect();
      const elementRect = selectedElement.getBoundingClientRect();
      
      // Check if element is not fully visible (partially or completely outside)
      const isFullyVisible = elementRect.top >= containerRect.top && elementRect.bottom <= containerRect.bottom;
      
      if (!isFullyVisible) {
        // Check if element is above the visible area
        if (elementRect.bottom < containerRect.top) {
          setStickyPosition('top');
        }
        // Check if element is below the visible area
        else if (elementRect.top > containerRect.bottom) {
          setStickyPosition('bottom');
        }
        // Element is partially visible - determine which side to stick to
        else if (elementRect.top < containerRect.top) {
          setStickyPosition('top');
        }
        else if (elementRect.bottom > containerRect.bottom) {
          setStickyPosition('bottom');
        }
      } else {
        setStickyPosition('none');
      }
    };

    // Initial check
    handleScroll();
    
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [selectedUnitId]);

  // Update selected unit data when selection changes (unit-based filtering)
  useEffect(() => {
    if (selectedUnitId) {
      // Determine eligible units: any piece matches current facet selection (respecting type and infoTypes)
      const eligibleUnits = new Set<string>();
      dataset.forEach((video) => {
        (video.pieces || []).forEach((p) => {
          if (selectedType !== "All" && p.content_type !== selectedType) return;
          if (!infoTypes.includes(p.content_type)) return;
          if (pieceMatchesFacet(p)) eligibleUnits.add(p.unit_id);
        });
      });

      // Aggregate ALL pieces for eligible units (respecting type/infoTypes only)
      const byUnit = new Map<string, { videos: Set<number>; samples: typeof allPieces }>();
      dataset.forEach((video, idx) => {
        (video.pieces || []).forEach((p) => {
          if (!eligibleUnits.has(p.unit_id)) return;
          if (selectedType !== "All" && p.content_type !== selectedType) return;
          if (!infoTypes.includes(p.content_type)) return;
          const e = byUnit.get(p.unit_id) || { videos: new Set<number>(), samples: [] as typeof allPieces };
          e.videos.add(idx);
          e.samples.push(p);
          byUnit.set(p.unit_id, e);
        });
      });

      const unitData = byUnit.get(selectedUnitId);
      if (unitData) {
        setSelectedUnitData({ unitId: selectedUnitId, info: unitData });
      }
    } else {
      setSelectedUnitData(null);
    }
  }, [selectedUnitId, dataset, pieceMatchesFacet, selectedType, infoTypes]);

  // Function to scroll to the original selected unit
  const scrollToOriginalUnit = () => {
    if (selectedUnitRef.current && scrollContainerRef.current) {
      const container = scrollContainerRef.current;
      const selectedElement = selectedUnitRef.current;
      
      const scrollTop = selectedElement.offsetTop - container.offsetTop - 
                       (container.clientHeight / 2) + (selectedElement.clientHeight / 2);
      
      container.scrollTo({
        top: Math.max(0, scrollTop),
        behavior: 'smooth'
      });
    }
  };
  return (
    <section className="rounded-xl border border-neutral-200 dark:border-neutral-800 p-4 space-y-3 min-w-0 min-h-0 h-full flex flex-col relative">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">{selectedType === "All" ? "All Information Units" : selectedType}</h3>
        <span className="text-sm text-neutral-500 bg-neutral-100 dark:bg-neutral-800 px-2 py-1 rounded-full">
          {(() => {
            // unit-based eligibility
            const eligibleUnits = new Set<string>();
            dataset.forEach((video) => {
              (video.pieces || []).forEach((p) => {
                if (selectedType !== "All" && p.content_type !== selectedType) return;
                if (!infoTypes.includes(p.content_type)) return;
                if (pieceMatchesFacet(p)) eligibleUnits.add(p.unit_id);
              });
            });
            return eligibleUnits.size;
          })()}
        </span>
      </div>
      <div ref={scrollContainerRef} className="space-y-2 flex-1 overflow-y-auto pr-1 min-w-0">
        {(() => {
          // Determine eligible units by facet selection
          const eligibleUnits = new Set<string>();
          dataset.forEach((video) => {
            (video.pieces || []).forEach((p) => {
              if (selectedType !== "All" && p.content_type !== selectedType) return;
              if (!infoTypes.includes(p.content_type)) return;
              if (pieceMatchesFacet(p)) eligibleUnits.add(p.unit_id);
            });
          });
          // Aggregate ALL pieces for eligible units (respecting type/infoTypes only)
          const byUnit = new Map<string, { videos: Set<number>; samples: typeof allPieces }>();
          dataset.forEach((video, idx) => {
            (video.pieces || []).forEach((p) => {
              if (!eligibleUnits.has(p.unit_id)) return;
              if (selectedType !== "All" && p.content_type !== selectedType) return;
              if (!infoTypes.includes(p.content_type)) return;
              const e = byUnit.get(p.unit_id) || { videos: new Set<number>(), samples: [] as typeof allPieces };
              e.videos.add(idx);
              e.samples.push(p);
              byUnit.set(p.unit_id, e);
            });
          });
          const rows = Array.from(byUnit.entries())
            // Sort by total occurrences (mentions) across all videos
            .sort((a, b) => b[1].samples.length - a[1].samples.length)
            .map(([unitId, info]) => {
              const first = info.samples[0];
              
              // Calculate contexts for this unit
              const buildContextKey = (p: DatasetVideo["pieces"][number]) => {
                const labels = p.labels || {} as Record<string, string[]>;
                const facetToLabelKey = (facet: Facet) =>
                  facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
                const keys = facets.length
                  ? facets.map(facetToLabelKey)
                  : Object.keys(labels).filter((k) => k.startsWith("context_"));
                const entries = keys
                  .map((k) => {
                    const rawVals = (labels[k] || []) as string[];
                    const vals = rawVals.map((v) => (v ?? "").trim());
                    const clean = vals.filter(Boolean);
                    const normalized = (clean.length ? clean : ["na"]).sort().join("|");
                    return [k, normalized] as const;
                  })
                  .sort((a, b) => a[0].localeCompare(b[0]));
                return entries.map(([k, v]) => `${k}=${v}`).join(";");
              };

              const contexts = new Set<string>();
              info.samples.forEach((p) => {
                const contextKey = buildContextKey(p);
                if (contextKey) contexts.add(contextKey);
              });

              // Aggregate contexts aligned by facets: facet -> set of values
              const facetToValues = new Map<string, Set<string>>();
              info.samples.forEach((p) => {
                const labels = (p.labels || {}) as Record<string, string[]>;
                const facetToLabelKey = (facet: Facet) =>
                  facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
                const keys = facets.length
                  ? facets.map(facetToLabelKey)
                  : Object.keys(labels).filter((k) => k.startsWith("context_"));
                keys.forEach((k) => {
                    const rawVals = (labels[k] || []) as string[];
                    const vals = rawVals.map((v) => (v ?? "").trim());
                    const set = facetToValues.get(k) || new Set<string>();
                    const clean = vals.filter(Boolean);
                    if (clean.length === 0) set.add("na");
                    else clean.forEach((v) => set.add(v));
                    facetToValues.set(k, set);
                  });
              });

              return (
                <div
                  key={unitId}
                  ref={selectedUnitId === unitId ? selectedUnitRef : null}
                  role="button"
                  tabIndex={0}
                  onClick={() => onSelect(unitId)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(unitId); } }}
                  className={`w-full text-left rounded-xl border p-3 transition-colors cursor-pointer ${
                    selectedUnitId === unitId
                      ? "border-blue-300 dark:border-blue-700 bg-blue-100 dark:bg-blue-900"
                      : "border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium">{first?.content || unitId}</div>
                      <div className="mt-1 text-xs text-neutral-500">{first?.content_type}</div>
                      <FacetValuesCompact
                        unitId={unitId}
                        facetCatalog={facetCatalog}
                        facetToValues={facetToValues}
                        selectedFacets={selectedFacets}
                      />
                    </div>
                    <ThumbnailMentions 
                      mentions={info.samples.length}
                      videoCount={info.videos.size}
                      unitId={unitId}
                      className="ml-3 flex-shrink-0"
                      contentType={first?.content_type}
                    />
                  </div>
                </div>
              );
            });
          return rows.length ? rows : <div className="text-sm text-neutral-500">No units for this type.</div>;
        })()}
      </div>
      
      {/* Sticky selected unit - positioned relative to the section */}
      {selectedUnitData && stickyPosition !== 'none' && (() => {
        // Calculate contexts for sticky unit
        const buildContextKey = (p: DatasetVideo["pieces"][number]) => {
          const labels = p.labels || {} as Record<string, string[]>;
          const facetToLabelKey = (facet: Facet) =>
            facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
          const keys = facets.length
            ? facets.map(facetToLabelKey)
            : Object.keys(labels).filter((k) => k.startsWith("context_"));
          const entries = keys
            .map((k) => {
              const rawVals = (labels[k] || []) as string[];
              const vals = rawVals.map((v) => (v ?? "").trim());
              const clean = vals.filter(Boolean);
              const normalized = (clean.length ? clean : ["na"]).sort().join("|");
              return [k, normalized] as const;
            })
            .sort((a, b) => a[0].localeCompare(b[0]));
          return entries.map(([k, v]) => `${k}=${v}`).join(";");
        };

        const contexts = new Set<string>();
        selectedUnitData.info.samples.forEach((p: DatasetVideo["pieces"][number]) => {
          const contextKey = buildContextKey(p);
          if (contextKey) contexts.add(contextKey);
        });

        return (
          <div className={`absolute left-4 right-4 ${stickyPosition === 'top' ? 'top-14' : 'bottom-4'} z-10`}>
            <div 
              className="w-full text-left rounded-xl border border-blue-300 dark:border-blue-700 bg-blue-100 dark:bg-blue-900 p-3 shadow-lg cursor-pointer hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
              onClick={scrollToOriginalUnit}
              title="Click to scroll to original unit"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="font-medium">
                    {selectedUnitData.info.samples[0]?.content || selectedUnitData.unitId}
                  </div>
                  <div className="mt-1 text-xs text-neutral-500">
                    {selectedUnitData.info.samples[0]?.content_type}
                  </div>
                  <FacetValuesCompact
                    unitId={selectedUnitData.unitId}
                    facetCatalog={facetCatalog}
                    facetToValues={(function(){
                      const map = new Map<string, Set<string>>();
                      selectedUnitData.info.samples.forEach((p: DatasetVideo["pieces"][number]) => {
                        const labels = (p.labels || {}) as Record<string, string[]>;
                        const facetToLabelKey = (facet: Facet) =>
                          facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
                        const keys = facets.length
                          ? facets.map(facetToLabelKey)
                          : Object.keys(labels).filter((k) => k.startsWith("context_"));
                        keys.forEach((k) => {
                            const rawVals = (labels[k] || []) as string[];
                            const vals = rawVals.map((v) => (v ?? "").trim());
                            const set = map.get(k) || new Set<string>();
                            const clean = vals.filter(Boolean);
                            if (clean.length === 0) set.add("na");
                            else clean.forEach((v) => set.add(v));
                            map.set(k, set);
                          });
                      });
                      return map;
                    })()}
                    selectedFacets={selectedFacets}
                    collapsible={true}
                    defaultCollapsed={true}
                  />
                </div>
                <ThumbnailMentions 
                  mentions={selectedUnitData.info.samples.length}
                  videoCount={selectedUnitData.info.videos.size}
                  unitId={selectedUnitData.unitId}
                  className="ml-3 flex-shrink-0"
                  contentType={selectedUnitData.info.samples[0]?.content_type}
                />
              </div>
            </div>
          </div>
        );
      })()}
    </section>
  );
}

