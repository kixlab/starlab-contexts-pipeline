"use client";

import { useState } from "react";
import type { TreeBuildConfig } from "../utils/treeBuilder";
import type { DatasetStatus } from "../types";

type ToolbarProps = {
  datasetOptions: { label: string }[];
  selectedLabel: string | null;
  onSelectDataset: (label: string) => void;
  datasetStatus: DatasetStatus;
  loadProgress: number | null;
  computeStatus: string;
  config: TreeBuildConfig;
  onConfigChange: (config: TreeBuildConfig) => void;
  onBuild: () => void;
  isBuilding: boolean;
  configDirty: boolean;
  treeSummary: { nodeCount: number; edgeCount: number; rootCount: number } | null;
  visibleCount: number;
  visibleEdgeCount: number;
};

export function Toolbar({
  datasetOptions,
  selectedLabel,
  onSelectDataset,
  datasetStatus,
  loadProgress,
  computeStatus,
  config,
  onConfigChange,
  onBuild,
  isBuilding,
  configDirty,
  treeSummary,
  visibleCount,
  visibleEdgeCount,
}: ToolbarProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const busy = isBuilding || datasetStatus === "loading" || datasetStatus === "computing";
  const canBuild = selectedLabel && datasetStatus === "loaded" && !busy;

  return (
    <header className="flex-shrink-0 border-b border-neutral-800 bg-neutral-950 px-4 py-2">
      {/* Main row */}
      <div className="flex items-center gap-3 text-sm">
        {/* Dataset selector */}
        <select
          value={selectedLabel ?? ""}
          onChange={(e) => e.target.value && onSelectDataset(e.target.value)}
          className="h-7 px-2 rounded bg-neutral-800 border border-neutral-700 text-white text-xs"
        >
          <option value="">Select dataset…</option>
          {datasetOptions.map((o) => (
            <option key={o.label} value={o.label}>{o.label}</option>
          ))}
        </select>

        {/* Loading indicator */}
        {datasetStatus === "loading" && (
          <span className="text-blue-400 text-xs flex items-center gap-1">
            <Spinner /> Loading{loadProgress != null ? ` ${Math.round(loadProgress * 100)}%` : "…"}
          </span>
        )}
        {datasetStatus === "computing" && (
          <span className="text-blue-400 text-xs flex items-center gap-1">
            <Spinner /> {computeStatus || "Computing…"}
          </span>
        )}

        {/* Separator */}
        {datasetStatus === "loaded" && (
          <>
            <div className="w-px h-4 bg-neutral-700" />

            {/* Quality metric toggle */}
            <div className="flex items-center gap-1 text-xs">
              <button
                onClick={() => onConfigChange({ ...config, qualityMetric: "D" })}
                className={`px-1.5 py-0.5 rounded ${config.qualityMetric === "D" ? "bg-blue-600 text-white" : "text-neutral-400 hover:text-white"}`}
              >
                D
              </button>
              <button
                onClick={() => onConfigChange({ ...config, qualityMetric: "eta2" })}
                className={`px-1.5 py-0.5 rounded ${config.qualityMetric === "eta2" ? "bg-blue-600 text-white" : "text-neutral-400 hover:text-white"}`}
              >
                η²
              </button>
            </div>

            {/* Tau slider */}
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
              τ
              <input
                type="range"
                min={0.5}
                max={1}
                step={0.01}
                value={config.refinementThreshold}
                onChange={(e) => onConfigChange({ ...config, refinementThreshold: parseFloat(e.target.value) })}
                className="w-20 h-1 accent-blue-500"
              />
              <span className="text-neutral-300 font-mono w-8">{config.refinementThreshold.toFixed(2)}</span>
            </label>

            {/* Advanced toggle */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-neutral-500 hover:text-neutral-300"
            >
              {showAdvanced ? "▾ Adv" : "▸ Adv"}
            </button>

            {/* Build button */}
            <button
              onClick={onBuild}
              disabled={!canBuild}
              className={`h-7 px-3 rounded text-xs font-medium transition-colors ${
                canBuild && configDirty
                  ? "bg-blue-600 hover:bg-blue-500 text-white"
                  : canBuild
                    ? "bg-neutral-700 hover:bg-neutral-600 text-neutral-300"
                    : "bg-neutral-800 text-neutral-600 cursor-not-allowed"
              }`}
            >
              {isBuilding ? "Building…" : configDirty ? "Build" : "Rebuild"}
            </button>
          </>
        )}

        {/* Right side: summary stats */}
        {treeSummary && (
          <div className="ml-auto flex items-center gap-3 text-xs text-neutral-500">
            <span>{visibleCount}/{treeSummary.nodeCount} nodes</span>
            <span>{visibleEdgeCount} edges</span>
            <span className="text-blue-400">{treeSummary.rootCount} roots</span>
          </div>
        )}
      </div>

      {/* Advanced row (collapsible) */}
      {showAdvanced && datasetStatus === "loaded" && (
        <div className="flex items-center gap-4 mt-2 text-xs text-neutral-400">
          <label className="flex items-center gap-1.5">
            Max K
            <input
              type="number"
              min={1}
              max={5}
              value={config.maxK}
              onChange={(e) => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v >= 1) onConfigChange({ ...config, maxK: Math.min(v, 5) });
              }}
              className="w-12 h-6 px-1.5 bg-neutral-800 border border-neutral-700 rounded text-white font-mono text-xs text-right"
            />
          </label>
          <label className="flex items-center gap-1.5">
            Min stability
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={config.stabilityThreshold}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v)) onConfigChange({ ...config, stabilityThreshold: Math.max(0, Math.min(1, v)) });
              }}
              className="w-14 h-6 px-1.5 bg-neutral-800 border border-neutral-700 rounded text-white font-mono text-xs text-right"
            />
          </label>
        </div>
      )}
    </header>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin" width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}
