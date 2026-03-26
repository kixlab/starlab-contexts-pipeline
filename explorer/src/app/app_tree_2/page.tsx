"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import type { DatasetVideo, Facet } from "@/app/types";
import { normalizeBundle } from "@/app/utils/bundleConverter";
import { DATASET_OPTIONS, deriveInfoTypesFromDataset } from "@/app/constants/datasetConfig";
import {
  enumerateSchemas,
  buildTreeFromCachedSchemas,
  DEFAULT_TREE_CONFIG,
  type CachedSchemas,
  type RefinementTree,
  type TreeBuildConfig,
} from "./utils/treeBuilder";
import { computeDOrder, computePositions } from "./utils/xyLayout";
import { computeHighlightedNodeIds, computeVisibleEdges } from "./utils/visibility";
import { Toolbar } from "./components/Toolbar";
import { TreeCanvas } from "./components/TreeCanvas";
import { InspectionPanel } from "./components/InspectionPanel";
import type { DatasetStatus } from "./types";

const LOADABLE_OPTIONS = DATASET_OPTIONS.filter((o) => !!o.bundleUrl);
const EMPTY_TREE: RefinementTree = { nodes: [], edges: [], roots: [] };

function configFromParams(params: URLSearchParams): { label: string | null; config: TreeBuildConfig } {
  const label = params.get("dataset") || null;
  const maxK = params.has("maxK") ? Math.max(1, Math.min(5, Number(params.get("maxK")))) : DEFAULT_TREE_CONFIG.maxK;
  const tau = params.has("tau") ? Math.max(0.5, Math.min(1, Number(params.get("tau")))) : DEFAULT_TREE_CONFIG.refinementThreshold;
  const stability = params.has("stability") ? Math.max(0, Math.min(1, Number(params.get("stability")))) : DEFAULT_TREE_CONFIG.stabilityThreshold;
  return {
    label: label && LOADABLE_OPTIONS.some((o) => o.label === label) ? label : null,
    config: { ...DEFAULT_TREE_CONFIG, maxK: isNaN(maxK) ? DEFAULT_TREE_CONFIG.maxK : maxK, refinementThreshold: isNaN(tau) ? DEFAULT_TREE_CONFIG.refinementThreshold : tau, stabilityThreshold: isNaN(stability) ? DEFAULT_TREE_CONFIG.stabilityThreshold : stability },
  };
}

export default function AppTree2() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // --- Initialise from URL params ---
  const initial = useMemo(() => configFromParams(searchParams), []); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Dataset state ---
  const [selectedLabel, setSelectedLabel] = useState<string | null>(initial.label);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>("idle");
  const [loadProgress, setLoadProgress] = useState<number | null>(null);
  const [dataset, setDataset] = useState<DatasetVideo[]>([]);
  const [facets, setFacets] = useState<Facet[]>([]);
  const [infoTypes, setInfoTypes] = useState<string[] | null>(null);

  // --- Tree config + build ---
  const [config, setConfig] = useState<TreeBuildConfig>(initial.config);
  const [lastBuiltConfig, setLastBuiltConfig] = useState<TreeBuildConfig | null>(null);
  const [tree, setTree] = useState<RefinementTree>(EMPTY_TREE);
  const [computeStatus, setComputeStatus] = useState("");
  const [isBuilding, setIsBuilding] = useState(false);

  // Phase 1 cache
  const cachedRef = useRef<CachedSchemas | null>(null);
  const cachedLabelRef = useRef<string | null>(null);
  const cachedMaxKRef = useRef<number>(DEFAULT_TREE_CONFIG.maxK);
  const loadingRef = useRef(false);

  // --- Selection ---
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [selectedEdgeKey, setSelectedEdgeKey] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Derived: dirty flag
  const configDirty = lastBuiltConfig === null || JSON.stringify(config) !== JSON.stringify(lastBuiltConfig);

  // --- Sync config to URL ---
  useEffect(() => {
    const params = new URLSearchParams();
    if (selectedLabel) params.set("dataset", selectedLabel);
    if (config.maxK !== DEFAULT_TREE_CONFIG.maxK) params.set("maxK", String(config.maxK));
    if (config.refinementThreshold !== DEFAULT_TREE_CONFIG.refinementThreshold) params.set("tau", String(config.refinementThreshold));
    if (config.stabilityThreshold !== DEFAULT_TREE_CONFIG.stabilityThreshold) params.set("stability", String(config.stabilityThreshold));
    const qs = params.toString();
    const newUrl = qs ? `${pathname}?${qs}` : pathname;
    router.replace(newUrl, { scroll: false });
  }, [selectedLabel, config, pathname, router]);

  // --- Global order (stable, computed once after build) ---
  const [globalOrder, setGlobalOrder] = useState<Map<string, number>>(new Map());
  const [allBands, setAllBands] = useState<number[]>([]);

  const highlightedNodeIds = useMemo(
    () => computeHighlightedNodeIds(selectedNodeIds, tree.edges),
    [selectedNodeIds, tree.edges],
  );

  const visibleEdges = useMemo(
    () => computeVisibleEdges(selectedNodeIds, tree.edges),
    [selectedNodeIds, tree.edges],
  );

  // --- Layout positions ---
  const [containerSize, setContainerSize] = useState({ w: 800, h: 600 });

  const layout = useMemo(
    () => computePositions(tree.nodes, tree.edges, globalOrder, containerSize.w, containerSize.h, allBands),
    [tree.nodes, tree.edges, globalOrder, containerSize, allBands],
  );

  // --- Dataset loading ---
  const fetchDataset = useCallback(async (label: string) => {
    if (loadingRef.current) return;
    const option = LOADABLE_OPTIONS.find((o) => o.label === label);
    if (!option?.bundleUrl) return;

    loadingRef.current = true;
    setDatasetStatus("loading");
    setLoadProgress(null);
    setTree(EMPTY_TREE);
    setSelectedNodeIds([]);
    setSelectedEdgeKey(null);
    setLastBuiltConfig(null);
    cachedRef.current = null;
    cachedLabelRef.current = null;

    try {
      const response = await fetch(option.bundleUrl);
      const contentLength = response.headers.get("Content-Length");
      const total = contentLength ? parseInt(contentLength, 10) : 0;

      let raw: unknown;
      if (response.body && total > 0) {
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;
        let lastMs = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
            received += value.length;
            const now = Date.now();
            if (now - lastMs > 80) { lastMs = now; setLoadProgress(received / total); }
          }
        }
        const all = new Uint8Array(received);
        let pos = 0;
        for (const c of chunks) { all.set(c, pos); pos += c.length; }
        raw = JSON.parse(new TextDecoder().decode(all));
      } else {
        raw = await response.json();
      }

      await new Promise<void>((r) => setTimeout(r, 0));
      const bundle = normalizeBundle(raw as Record<string, unknown>);
      const ds = ((bundle as any)?.labeled_dataset ?? []) as DatasetVideo[];
      const fs = ((bundle as any)?.context_schema ?? []) as Facet[];
      const it = option.infoTypes === null ? deriveInfoTypesFromDataset(ds) : option.infoTypes;

      setDataset(ds);
      setFacets(fs);
      setInfoTypes(it);
      setDatasetStatus("loaded");
      setLoadProgress(null);
      setComputeStatus("");
    } catch (err) {
      console.error("Failed to load dataset:", label, err);
      setDatasetStatus("error");
    } finally {
      loadingRef.current = false;
      setLoadProgress(null);
    }
  }, []);

  const handleSelectDataset = useCallback((label: string) => {
    if (label === selectedLabel) return;
    setSelectedLabel(label);
  }, [selectedLabel]);

  useEffect(() => {
    if (!selectedLabel) return;
    const needs =
      cachedLabelRef.current !== selectedLabel ||
      datasetStatus === "idle" ||
      datasetStatus === "error";
    if (needs) fetchDataset(selectedLabel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLabel]);

  // --- Build ---
  const handleBuild = useCallback(async () => {
    if (!selectedLabel || loadingRef.current || datasetStatus !== "loaded") return;

    const cfg = config;
    const needsEnum =
      !cachedRef.current ||
      cachedLabelRef.current !== selectedLabel ||
      cfg.maxK !== cachedMaxKRef.current;

    setIsBuilding(true);
    try {
      if (needsEnum) {
        setDatasetStatus("computing");
        const cached = await enumerateSchemas(
          dataset, facets, infoTypes, cfg.maxK,
          (step, done, total) => setComputeStatus(`${step}: ${done}/${total}`),
        );
        if (!cached) { setDatasetStatus("loaded"); setIsBuilding(false); return; }
        cachedRef.current = cached;
        cachedLabelRef.current = selectedLabel;
        cachedMaxKRef.current = cfg.maxK;
        setDatasetStatus("loaded");
        setComputeStatus("");
      }

      // Derive all possible compactness bands from cached schemas
      const bandSet = new Set<number>();
      for (const n of cachedRef.current!.allNodes) bandSet.add(n.size);
      setAllBands(Array.from(bandSet).sort((a, b) => a - b));

      const newTree = buildTreeFromCachedSchemas(cachedRef.current!, cfg);
      setTree(newTree);
      setLastBuiltConfig(cfg);
      setGlobalOrder(computeDOrder(newTree));
      const ids = new Set(newTree.nodes.map((n) => n.id));
      setSelectedNodeIds((prev) => prev.filter((id) => ids.has(id)));
      setSelectedEdgeKey((prev) => {
        if (!prev) return null;
        return newTree.edges.some((e) => `${e.sourceId}->${e.targetId}` === prev) ? prev : null;
      });
    } finally {
      setIsBuilding(false);
    }
  }, [config, selectedLabel, datasetStatus, dataset, facets, infoTypes]);

  // --- Tree summary ---
  const treeSummary = useMemo(
    () => tree.nodes.length > 0
      ? { nodeCount: tree.nodes.length, edgeCount: tree.edges.length, rootCount: tree.roots.length }
      : null,
    [tree],
  );

  // --- Scroll-to-node ---
  const [scrollToNodeId, setScrollToNodeId] = useState<string | null>(null);

  // --- Selection handlers ---
  const selectNode = useCallback((nodeId: string) => {
    setSelectedEdgeKey(null);
    setSelectedNodeIds((prev) => {
      if (prev.includes(nodeId)) return prev.filter((id) => id !== nodeId);
      if (prev.length >= 2) return [prev[1], nodeId];
      return [...prev, nodeId];
    });
  }, []);

  const handleNodeClick = useCallback((nodeId: string, shiftKey: boolean) => {
    if (shiftKey) {
      selectNode(nodeId);
    } else {
      // Normal click: select single node
      setSelectedEdgeKey(null);
      setSelectedNodeIds([nodeId]);
    }
  }, [selectNode]);

  const handleCompareClick = useCallback((nodeId: string) => {
    selectNode(nodeId);
  }, [selectNode]);

  const handleNodeClickFromPanel = useCallback((nodeId: string) => {
    setSelectedEdgeKey(null);
    setSelectedNodeIds([nodeId]);
    setScrollToNodeId(nodeId);
    setTimeout(() => setScrollToNodeId(null), 600);
  }, []);

  const handleEdgeClick = useCallback((edgeKey: string | null) => {
    setSelectedEdgeKey(edgeKey);
    if (edgeKey) {
      const [srcId, tgtId] = edgeKey.split('->');
      setSelectedNodeIds((prev) => {
        if (prev.includes(srcId)) return [srcId];
        if (prev.includes(tgtId)) return [tgtId];
        return [srcId];
      });
    }
  }, []);

  const handleDeselectAll = useCallback(() => {
    setSelectedNodeIds([]);
    setSelectedEdgeKey(null);
  }, []);

  const validPiecesCount = cachedRef.current?.validPiecesCount ?? 0;
  const numUniqueUnits = cachedRef.current?.numUniqueUnits ?? 0;

  return (
    <div className="flex h-screen bg-neutral-950 text-white overflow-hidden">
      {/* Main column: toolbar + canvas */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center gap-2 px-4 py-1 border-b border-neutral-800">
          <Link href="/" className="text-neutral-400 hover:text-white text-xs transition-colors">
            ← Back
          </Link>
          <span className="text-sm font-semibold">Schema Refinement Tree</span>
        </div>

        <Toolbar
          datasetOptions={LOADABLE_OPTIONS.map((o) => ({ label: o.label }))}
          selectedLabel={selectedLabel}
          onSelectDataset={handleSelectDataset}
          datasetStatus={datasetStatus}
          loadProgress={loadProgress}
          computeStatus={computeStatus}
          config={config}
          onConfigChange={setConfig}
          onBuild={handleBuild}
          isBuilding={isBuilding}
          configDirty={configDirty}
          treeSummary={treeSummary}
          visibleCount={tree.nodes.length}
          visibleEdgeCount={tree.edges.length}
        />

        <div className="flex-1 min-h-0 relative">
          {datasetStatus === "loaded" ? (
            <TreeCanvas
              tree={tree}
              visibleEdges={visibleEdges}
              positions={layout.positions}
              bandRows={layout.bandRows}
              rowHeight={layout.rowHeight}
              numUniqueUnits={numUniqueUnits}
              selectedNodeIds={selectedNodeIds}
              selectedEdgeKey={selectedEdgeKey}
              hoveredNodeId={hoveredNodeId}
              highlightedNodeIds={highlightedNodeIds}
              onNodeClick={handleNodeClick}
              onCompareClick={handleCompareClick}
              onEdgeClick={handleEdgeClick}
              onNodeHover={setHoveredNodeId}
              onDeselectAll={handleDeselectAll}
              onContainerResize={setContainerSize}
              scrollToNodeId={scrollToNodeId}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-neutral-500 text-sm">
              {!selectedLabel
                ? "Select a dataset to begin"
                : datasetStatus === "error"
                  ? "Failed to load dataset"
                  : ""}
            </div>
          )}
        </div>
      </div>

      {/* Right: inspection panel */}
      <InspectionPanel
        tree={tree}
        selectedNodeIds={selectedNodeIds}
        selectedEdgeKey={selectedEdgeKey}
        dataset={dataset}
        infoTypes={infoTypes}
        numUniqueUnits={numUniqueUnits}
        validPiecesCount={validPiecesCount}
        onNodeClick={handleNodeClickFromPanel}
        onClose={handleDeselectAll}
      />
    </div>
  );
}
