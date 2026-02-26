"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { FacetPanel } from "./components/FacetPanel";
import { StatsPanel } from "./components/StatsPanel";
import { UnitsList } from "./components/UnitsList";
import { DetailsPanel } from "./components/DetailsPanel";
import type { DatasetVideo, Facet } from "@/app/types";
import { normalizeBundle } from "@/app/utils/bundleConverter";
import { GRID_TEMPLATE } from "@/app/constants/ui";
import { getSelectedDataset } from "@/app/constants/datasetConfig";

export default function Home() {
  // Get dataset from localStorage or use default
  const selectedDataset = getSelectedDataset();
  const [bundleUrl] = useState(selectedDataset.bundleUrl);
  const [availableInfoTypes] = useState<string[]>([...selectedDataset.infoTypes]);
  const [dataset, setDataset] = useState<DatasetVideo[]>([]);
  // Facets are now adjustable:
  // - `shownFacets` are the facets rendered in the FacetPanel / used for context computations
  // - `facetCandidates` are the pool you can add from (bundle: `facet_candidates`)
  const [shownFacets, setShownFacets] = useState<Facet[]>([]);
  const [facetCandidates, setFacetCandidates] = useState<Facet[]>([]);
  const [selectedUnitId, setSelectedUnitId] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<string>("All");
  // Information types are now adjustable:
  // - `selectedInfoTypes` drives filtering + coverage computations
  // - `availableInfoTypes` comes from dataset configuration (datasetConfig.ts)
  const [selectedInfoTypes, setSelectedInfoTypes] = useState<string[]>(() => {
    // Start minimal: Method - Instruction only (if present); otherwise fall back to first available.
    if (availableInfoTypes.includes("Method - Instruction")) return ["Method - Instruction"];
    return availableInfoTypes.length ? [availableInfoTypes[0]] : [];
  });
  // Per-row facet selection: at most one label per facet key
  const [selectedFacets, setSelectedFacets] = useState<Record<string, { label: string; title: string }>>({});
  const [denomMode, setDenomMode] = useState<'relative' | 'theoretical'>('relative');
  const [hoverTip, setHoverTip] = useState<null | { x: number; y: number; title: string; text: string }>(null);

  // Update selectedInfoTypes when availableInfoTypes changes (e.g., from config)
  useEffect(() => {
    // If current selection is invalid or availableInfoTypes is empty, reset selection
    const isValid = selectedInfoTypes.every((t) => availableInfoTypes.includes(t));
    if (!isValid || (availableInfoTypes.length === 0 && selectedInfoTypes.length > 0)) {
      if (availableInfoTypes.includes("Method - Instruction")) {
        setSelectedInfoTypes(["Method - Instruction"]);
      } else if (availableInfoTypes.length > 0) {
        setSelectedInfoTypes([availableInfoTypes[0]]);
      } else {
        setSelectedInfoTypes([]);
      }
    }
  }, [availableInfoTypes]);

  useEffect(() => {
    const load = async () => {
      // Load bundled file with { context_schema, facet_candidates, labeled_dataset }
      if (!bundleUrl) return;
      try {
        const bundle = normalizeBundle(await fetch(bundleUrl).then((r) => r.json()));
        if (bundle && typeof bundle === "object" && bundle.labeled_dataset) {
          setDataset(bundle.labeled_dataset as DatasetVideo[]);
          const contextSchema = (bundle.context_schema ?? []) as Facet[];
          const candidates = (bundle.facet_candidates ?? []) as Facet[];

          // Start minimal: show no facets by default (user will add from candidates).
          const initialCandidates = candidates.length ? candidates : contextSchema;
          setShownFacets([]);
          setFacetCandidates(initialCandidates);
        }
      } catch (error) {
        console.error("Failed to load bundle:", error);
      }
    };
    load();
  }, [bundleUrl]);

  const allPieces = useMemo(() => dataset.flatMap((v) => v.pieces || []), [dataset]);

  const unitIdToPieces = useMemo(() => {
    const map = new Map<string, { count: number; samples: typeof allPieces }>();
    for (const p of allPieces) {
      const entry = map.get(p.unit_id) || { count: 0, samples: [] as typeof allPieces };
      entry.count += 1;
      entry.samples.push(p);
      map.set(p.unit_id, entry);
    }
    return map;
  }, [allPieces]);

  const totalVideos = dataset.length;

  const normalizeFacetVals = (rawVals: string[] | undefined) => {
    const vals = (rawVals || []).map((v) => (v ?? "").trim()).filter(Boolean);
    // `na` is now an official value; treat missing/blank as `na` for robustness.
    return vals.length ? vals : ["na"];
  };

  const pieceMatchesFacet = (p: DatasetVideo["pieces"][number]) => {
    const entries = Object.entries(selectedFacets);
    if (entries.length === 0) return true;
    for (const [key, sel] of entries) {
      const vals = normalizeFacetVals((p.labels?.[key] || []) as string[]);
      if (!vals.includes(sel.label)) return false;
    }
    return true;
  };

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

  const filteredPieces = useMemo(() => allPieces.filter(pieceMatchesFacet), [allPieces, selectedFacets]);

  const coverageByType = useMemo(() => {
    // Build contexts: combinations of facet values across all context_* labels
    const buildContextKey = (p: DatasetVideo["pieces"][number]) => {
      const labels = (p.labels || {}) as Record<string, string[]>;
      // New datasets use facet ids as keys; old datasets used `context_*` keys.
      const facetKeys =
        shownFacets?.length
          ? shownFacets.map((f) => f.id).filter(Boolean)
          : Object.keys(labels).filter((k) => k.startsWith("context_"));

      const entries = facetKeys
        .map((k) => {
          const raw = (labels[k] || []) as string[];
          const normalized = normalizeFacetVals(raw);
          return [k, normalized.sort().join("|")] as const;
        })
        .sort((a, b) => a[0].localeCompare(b[0]));
      return entries.map(([k, v]) => `${k}=${v}`).join(";");
    };

    const contextToTypes = new Map<string, Set<string>>();
    for (const p of filteredPieces) {
      const key = buildContextKey(p);
      if (!key) continue;
      const set = contextToTypes.get(key) || new Set<string>();
      set.add(p.content_type);
      contextToTypes.set(key, set);
    }

    // Calculate theoretical total based on unselected facets
    const theoreticalTotal = shownFacets.reduce((acc, f) => {
      const fKey = f.id;
      const isFixed = Boolean(selectedFacets[fKey]);
      const len = f.vocabulary?.length || 0;
      return acc * (isFixed ? 1 : Math.max(1, len));
    }, 1);

    const relativeTotal = contextToTypes.size;
    const totalContexts = denomMode === 'relative' ? relativeTotal : theoreticalTotal;
    
    const emptyByType = Object.fromEntries(selectedInfoTypes.map((t) => [t, 0])) as Record<string, number>;
    if (totalContexts === 0) {
      return { totalCoverage: 0, byType: emptyByType };
    }

    const byType = Object.fromEntries(
      selectedInfoTypes.map((t) => {
        let count = 0;
        for (const set of contextToTypes.values()) {
          if (set.has(t)) count += 1;
        }
        return [t, Math.round((count / totalContexts) * 100)];
      })
    ) as Record<string, number>;

    // Total coverage: simple average of per-type coverage
    const sum = selectedInfoTypes.reduce((acc, t) => acc + (byType[t] || 0), 0);
    const avg = sum / (selectedInfoTypes.length || 1);
    return { totalCoverage: Math.round(avg), byType };
  }, [filteredPieces, denomMode, selectedFacets, shownFacets, selectedInfoTypes]);

  // Removed hovered coverage calculation

  // Get eligible units (same logic as UnitsList)
  const eligibleUnits = useMemo(() => {
    const units = new Set<string>();
    dataset.forEach((video) => {
      (video.pieces || []).forEach((p) => {
        if (selectedType !== "All" && p.content_type !== selectedType) return;
        if (!selectedInfoTypes.includes(p.content_type as any)) return;
        if (pieceMatchesFacet(p)) units.add(p.unit_id);
      });
    });
    return units;
  }, [dataset, selectedType, selectedFacets, selectedInfoTypes]);

  // Clear selected unit if it's no longer eligible
  useEffect(() => {
    if (selectedUnitId && !eligibleUnits.has(selectedUnitId)) {
      setSelectedUnitId(null);
    }
  }, [selectedUnitId, eligibleUnits]);

  const selectedPieces = useMemo(() => {
    if (!selectedUnitId) return [] as typeof allPieces;
    const pieces = unitIdToPieces.get(selectedUnitId)?.samples ?? [];
    const byFacets = pieces.filter(pieceMatchesFacet);
    if (selectedType === "All") return byFacets;
    return byFacets.filter((p) => p.content_type === selectedType);
  }, [selectedUnitId, unitIdToPieces, selectedType, selectedFacets]);

  // If the user removes the currently selected info type, fall back to "All".
  useEffect(() => {
    if (selectedType !== "All" && !selectedInfoTypes.includes(selectedType)) {
      setSelectedType("All");
    }
  }, [selectedType, selectedInfoTypes]);

  // If the user removes a facet, ensure we drop any selection for it.
  const removeFacetFromShown = (facet: Facet) => {
    const key = facet.id || `context_${facet.title.trim().toLowerCase().replace(/\s+/g, "_")}`;
    setShownFacets((prev) => prev.filter((f) => (f.id || f.title) !== (facet.id || facet.title)));
    setSelectedFacets((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const addFacetToShown = (facet: Facet) => {
    setShownFacets((prev) => {
      const prevIds = new Set(prev.map((f) => f.id || f.title));
      const id = facet.id || facet.title;
      if (prevIds.has(id)) return prev;
      return [...prev, facet];
    });
  };

  // Grid template comes from constants

  return (
    <div className="h-screen w-full overflow-hidden font-sans bg-white dark:bg-black text-black dark:text-white">
      <div className="absolute top-4 left-4 z-10">
        <Link
          href="/"
          className="px-3 py-1.5 text-sm bg-neutral-100 dark:bg-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-700 rounded border border-neutral-300 dark:border-neutral-600 transition-colors"
        >
          ← Back to Home
        </Link>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[var(--grid)] gap-4 p-4 h-full min-h-0" style={{ ['--grid' as string]: GRID_TEMPLATE, gridTemplateRows: '1fr' }}>
        {/* Column 1: Facets */}
        <FacetPanel
          facets={shownFacets}
          facetCandidates={facetCandidates}
          onRemoveFacet={removeFacetFromShown}
          onAddFacet={addFacetToShown}
          allPieces={allPieces}
          selectedFacets={selectedFacets}
          setSelectedFacets={setSelectedFacets}
          selectedType={selectedType}
          infoTypes={selectedInfoTypes}
          denomMode={denomMode}
          setDenomMode={setDenomMode}
        />

        {/* Column 2: Stats */}
        <StatsPanel
          totalVideos={totalVideos}
          videoCount={(function(){
            const s=new Set<number>();
            dataset.forEach((v,idx)=>{ if((v.pieces||[]).some(pieceMatchesFacet)) s.add(idx); });
            return s.size;
          })()}
          selectedFacets={selectedFacets}
          coverageByType={coverageByType}
          selectedType={selectedType}
          setSelectedType={setSelectedType}
          infoTypes={selectedInfoTypes}
          availableInfoTypes={availableInfoTypes}
          setInfoTypes={setSelectedInfoTypes}
        />

        {/* Column 3: Information Units */}
        <UnitsList
          dataset={dataset}
          allPieces={allPieces}
          facets={shownFacets}
          selectedType={selectedType}
          pieceMatchesFacet={pieceMatchesFacet}
          selectedUnitId={selectedUnitId}
          onSelect={(id)=>setSelectedUnitId(id)}
          infoTypes={selectedInfoTypes}
          selectedFacets={selectedFacets}
        />

        {/* Column 4: Details Panel */}
        <DetailsPanel pieces={selectedPieces} dataset={dataset} facets={shownFacets} selectedFacets={selectedFacets} onClear={()=>setSelectedUnitId(null)} infoTypes={selectedInfoTypes} />
      </div>
      {hoverTip && (
        <div
          className="fixed z-[1000] w-80 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 p-3 text-xs text-neutral-700 dark:text-neutral-300 shadow-2xl whitespace-pre-wrap"
          style={{ left: hoverTip.x + 12, top: hoverTip.y }}
        >
          <div className="font-medium mb-1">{hoverTip.title}</div>
          {hoverTip.text}
        </div>
      )}
    </div>
  );
}
