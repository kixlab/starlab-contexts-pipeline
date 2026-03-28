"use client";

import { useEffect, useState } from "react";
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
  onDownloadLog?: () => void;
  hasLogEntries?: boolean;
  buildTimestamp?: number | null;
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
  onDownloadLog,
  hasLogEntries,
  buildTimestamp,
}: ToolbarProps) {
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

            {/* Tau */}
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
              τ
              <input
                type="number"
                min={0.5}
                max={1}
                step={0.01}
                value={config.refinementThreshold}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v)) onConfigChange({ ...config, refinementThreshold: Math.max(0.5, Math.min(1, v)) });
                }}
                className="w-14 h-6 px-1.5 bg-neutral-800 border border-neutral-700 rounded text-white font-mono text-xs text-right"
              />
            </label>

            {/* Max K */}
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
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

            {/* Min stability */}
            <label className="flex items-center gap-1.5 text-xs text-neutral-400">
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

        {/* Right side: summary stats + log download */}
        {treeSummary && (
          <div className="ml-auto flex items-center gap-3 text-xs text-neutral-500">
            <span>{visibleCount}/{treeSummary.nodeCount} nodes</span>
            <span>{visibleEdgeCount} edges</span>
            <span className="text-blue-400">{treeSummary.rootCount} roots</span>
            {buildTimestamp && <ElapsedTimer since={buildTimestamp} />}
            {onDownloadLog && (
              <button
                onClick={onDownloadLog}
                disabled={!hasLogEntries}
                className={`h-6 px-2 rounded text-[10px] border transition-colors ${
                  hasLogEntries
                    ? "border-neutral-600 text-neutral-300 hover:bg-neutral-800"
                    : "border-neutral-800 text-neutral-700 cursor-not-allowed"
                }`}
              >
                ↓ Log
              </button>
            )}
          </div>
        )}
      </div>

    </header>
  );
}

function ElapsedTimer({ since }: { since: number }) {
  const [elapsed, setElapsed] = useState(() => Math.floor((Date.now() - since) / 1000));

  useEffect(() => {
    setElapsed(Math.floor((Date.now() - since) / 1000));
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - since) / 1000)), 1000);
    return () => clearInterval(id);
  }, [since]);

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");

  return <span className="font-mono text-neutral-400">{mm}:{ss}</span>;
}

function Spinner() {
  return (
    <svg className="animate-spin" width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}
