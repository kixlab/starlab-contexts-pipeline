"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TreeNode, TreeEdge, RefinementTree } from "../utils/treeBuilder";
import { computeEffectiveness } from "../utils/treeBuilder";
import { NODE_WIDTH, NODE_HEIGHT, MARGIN, type LayoutResult } from "../utils/xyLayout";
import type { NodePosition } from "../types";
import { facetIdToChipColor } from "@/app/app_analysis/utils/metricsUtils";

// Adaptive node height constants
const CHIP_H = 16;
const BASE_H = 36; // metrics line + padding

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type TreeCanvasProps = {
  tree: RefinementTree;
  positions: Map<string, NodePosition>;
  bandRows: LayoutResult["bandRows"];
  rowHeight: number;
  numUniqueUnits: number;
  selectedNodeIds: string[];
  selectedEdgeKey: string | null;
  hoveredNodeId: string | null;
  highlightedNodeIds: Set<string>;
  onNodeClick: (nodeId: string, shiftKey: boolean) => void;
  onCompareClick: (nodeId: string) => void;
  onEdgeClick: (edgeKey: string | null) => void;
  onNodeHover: (nodeId: string | null) => void;
  onDeselectAll: () => void;
  onContainerResize: (size: { w: number; h: number }) => void;
  scrollToNodeId?: string | null;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TreeCanvas({
  tree,
  positions,
  bandRows,
  rowHeight,
  numUniqueUnits,
  selectedNodeIds,
  selectedEdgeKey,
  hoveredNodeId,
  highlightedNodeIds,
  onNodeClick,
  onCompareClick,
  onEdgeClick,
  onNodeHover,
  onDeselectAll,
  onContainerResize,
  scrollToNodeId,
}: TreeCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  // Resize observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) onContainerResize({ w: rect.width, h: rect.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [onContainerResize]);

  // SVG dimensions from positions
  const svgWidth = useMemo(() => {
    let maxX = 800;
    for (const pos of positions.values()) {
      maxX = Math.max(maxX, pos.x + NODE_WIDTH / 2 + MARGIN.right);
    }
    return maxX;
  }, [positions]);

  const svgHeight = useMemo(() => {
    let maxY = 600;
    // Use max possible node height for bottom padding
    let maxH = NODE_HEIGHT;
    for (const n of tree.nodes) maxH = Math.max(maxH, BASE_H + n.facetSubset.length * CHIP_H);
    for (const pos of positions.values()) {
      maxY = Math.max(maxY, pos.y + maxH / 2 + MARGIN.bottom);
    }
    return maxY;
  }, [positions, tree.nodes]);

  // Layers for background stripes + labels (from layout)
  const layers = bandRows;

  // Scroll to node
  useEffect(() => {
    if (!scrollToNodeId) return;
    const pos = positions.get(scrollToNodeId);
    const el = containerRef.current;
    if (!pos || !el) return;
    el.scrollTo({
      left: pos.x - el.clientWidth / 2,
      top: pos.y - el.clientHeight / 2,
      behavior: "smooth",
    });
  }, [scrollToNodeId, positions]);


  // Adaptive node height: base for metrics + 16px per facet chip
  const nodeHeight = useCallback((n: TreeNode) => BASE_H + n.facetSubset.length * CHIP_H, []);

  // Pre-compute heights for edge routing
  const nodeHeightMap = useMemo(() => {
    const m = new Map<string, number>();
    for (const n of tree.nodes) m.set(n.id, nodeHeight(n));
    return m;
  }, [tree.nodes, nodeHeight]);

  // Children count (for display)
  const childrenCountMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const n of tree.nodes) map.set(n.id, 0);
    for (const e of tree.edges) map.set(e.sourceId, (map.get(e.sourceId) ?? 0) + 1);
    return map;
  }, [tree]);

  // Edge helpers
  const edgePath = useCallback((e: TreeEdge) => {
    const src = positions.get(e.sourceId);
    const tgt = positions.get(e.targetId);
    if (!src || !tgt) return "";
    const srcH = nodeHeightMap.get(e.sourceId) ?? NODE_HEIGHT;
    const tgtH = nodeHeightMap.get(e.targetId) ?? NODE_HEIGHT;
    const sy = src.y + srcH / 2;
    const ty = tgt.y - tgtH / 2;
    const midY = (sy + ty) / 2;
    return `M ${src.x} ${sy} C ${src.x} ${midY}, ${tgt.x} ${midY}, ${tgt.x} ${ty}`;
  }, [positions, nodeHeightMap]);

  const EDGE_COLORS: Record<string, string> = { add: "#60a5fa", restructure: "#facc15" };

  const handleBgClick = useCallback(() => {
    onEdgeClick(null);
    onDeselectAll();
  }, [onEdgeClick, onDeselectAll]);

  // Node fill based on effectiveness
  const nodeFill = (n: TreeNode) => {
    const eff = computeEffectiveness(n.D, numUniqueUnits);
    const t = eff / 100;
    return `rgb(${30 + t * 30},${30 + t * 60},${50 + t * 80})`;
  };

  if (tree.nodes.length === 0) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center text-neutral-500 text-sm">
        No schemas pass the current filters. Click Build to generate the tree.
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-auto">
      {/* Row stripes */}
      <div className="absolute inset-0 pointer-events-none" style={{ height: svgHeight, width: svgWidth }}>
        {layers.map((l, i) => (
          <div
            key={`stripe-${l.band}`}
            className={`absolute left-0 border-b border-neutral-700/40 ${i % 2 === 0 ? "bg-neutral-950" : "bg-neutral-800/25"}`}
            style={{ top: l.y - rowHeight / 2, height: rowHeight, width: "100%" }}
          />
        ))}
      </div>

      {/* SVG */}
      <svg width={svgWidth} height={svgHeight} className="relative" onClick={handleBgClick}>
        <defs>
          <marker id="t2-arrow-add" viewBox="0 0 10 8" refX="10" refY="4" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 4 L 0 8 z" fill="#60a5fa" />
          </marker>
          <marker id="t2-arrow-restructure" viewBox="0 0 10 8" refX="10" refY="4" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 4 L 0 8 z" fill="#facc15" />
          </marker>
          <marker id="t2-arrow-sel" viewBox="0 0 10 8" refX="10" refY="4" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 4 L 0 8 z" fill="#f59e0b" />
          </marker>
        </defs>

        {/* Edges */}
        {tree.edges.map((e) => {
          const key = `${e.sourceId}->${e.targetId}`;
          const isSel = key === selectedEdgeKey;
          const color = EDGE_COLORS[e.edgeType] ?? "#6b7280";
          const dash = e.edgeType === "restructure" ? "6 3" : "none";
          return (
            <g key={key}>
              <path d={edgePath(e)} fill="none" stroke="transparent" strokeWidth={12}
                className="cursor-pointer"
                onClick={(ev) => { ev.stopPropagation(); onEdgeClick(isSel ? null : key); }}
                onMouseEnter={(ev) => {
                  const rect = containerRef.current?.getBoundingClientRect();
                  if (rect) {
                    setTooltip({
                      x: ev.clientX - rect.left + (containerRef.current?.scrollLeft ?? 0),
                      y: ev.clientY - rect.top + (containerRef.current?.scrollTop ?? 0) - 10,
                      text: `[${e.edgeType}] Refinement=${Math.round(e.refinementScore * 100)}%`,
                    });
                  }
                }}
                onMouseLeave={() => setTooltip(null)}
              />
              <path d={edgePath(e)} fill="none"
                stroke={isSel ? "#f59e0b" : color}
                strokeWidth={isSel ? 2.5 : 1 + e.refinementScore * 2}
                strokeOpacity={isSel ? 1 : 0.7}
                strokeDasharray={isSel ? "6 3" : dash}
                markerEnd={`url(#t2-arrow-${isSel ? "sel" : e.edgeType})`}
                className="pointer-events-none"
              />
            </g>
          );
        })}

        {/* Nodes */}
        {tree.nodes.map((node) => {
          const pos = positions.get(node.id);
          if (!pos || pos.isDummy) return null;
          const isRoot = tree.roots.includes(node.id);
          const selIdx = selectedNodeIds.indexOf(node.id);
          const isSel = selIdx >= 0;
          const isHov = node.id === hoveredNodeId;
          const isHi = highlightedNodeIds.has(node.id);
          const numCh = childrenCountMap.get(node.id) ?? 0;
          const eff = computeEffectiveness(node.D, numUniqueUnits);
          const selColor = selIdx === 0 ? "#f59e0b" : selIdx === 1 ? "#22d3ee" : null;
          const nh = nodeHeight(node);

          return (
            <g
              key={node.id}
              transform={`translate(${pos.x - NODE_WIDTH / 2}, ${pos.y - nh / 2})`}
              className="cursor-pointer"
              onClick={(ev) => { ev.stopPropagation(); onNodeClick(node.id, ev.shiftKey); }}
              onMouseEnter={() => onNodeHover(node.id)}
              onMouseLeave={() => onNodeHover(null)}
            >
              {/* Highlight ring */}
              {isHi && !isSel && (
                <rect x={-4} y={-4} width={NODE_WIDTH + 8} height={nh + 8} rx={11}
                  fill="none" stroke="#60a5fa" strokeWidth={2} strokeDasharray="4 2" strokeOpacity={0.8} />
              )}

              {/* Background */}
              <rect width={NODE_WIDTH} height={nh} rx={8} fill={nodeFill(node)}
                stroke={selColor ?? (isHov ? "#525252" : isRoot ? "#3b82f6" : "#404040")}
                strokeWidth={isSel ? 2.5 : isRoot ? 2 : 1} />

              {/* Root bar */}
              {isRoot && <rect width={NODE_WIDTH} height={3} rx={1.5} fill="#3b82f6" />}

              {/* Compare icon */}
              <g className={`${isHov ? "opacity-80" : "opacity-0"} transition-opacity`}
                onClick={(ev) => { ev.stopPropagation(); onCompareClick(node.id); }}>
                <circle cx={10} cy={10} r={8} fill="#171717" stroke="#525252" strokeWidth={0.5} />
                <text x={10} y={14} textAnchor="middle" className="text-[9px] font-bold" fill="#a3a3a3">⊕</text>
              </g>

              {/* Selection badge */}
              {isSel && (
                <>
                  <circle cx={NODE_WIDTH - 2} cy={-2} r={9} fill={selColor!} />
                  <text x={NODE_WIDTH - 2} y={2} textAnchor="middle" className="text-[10px] font-bold" fill="#000">
                    {selIdx + 1}
                  </text>
                </>
              )}

              {/* Facet chips — stacked vertically */}
              <g transform="translate(6, 8)">
                {node.facetSubset.map((facet, fi) => {
                  const c = facetIdToChipColor(facet.id);
                  const chipW = Math.min(NODE_WIDTH - 12, Math.max(30, facet.title.length * 5.5 + 8));
                  return (
                    <g key={facet.id} transform={`translate(0, ${fi * CHIP_H})`}>
                      <rect width={chipW} height={14} rx={3} fill={c + "30"} stroke={c + "60"} strokeWidth={0.5} />
                      <text x={chipW / 2} y={10} textAnchor="middle" className="text-[8px] font-medium" fill={c}>{facet.title}</text>
                    </g>
                  );
                })}
              </g>

              {/* Metrics */}
              <text x={NODE_WIDTH / 2} y={nh - 14} textAnchor="middle" className="text-[9px]" fill="#a3a3a3">
                Size={node.size} · Eff={eff.toFixed(0)}%
              </text>

              {/* Bottom: children count */}
              {numCh > 0 && (
                <text x={NODE_WIDTH / 2} y={nh - 4} textAnchor="middle" className="text-[8px]" fill="#737373">
                  {numCh} children
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div className="absolute pointer-events-none bg-neutral-800 border border-neutral-700 rounded-md px-3 py-2 text-xs text-neutral-200 whitespace-pre-wrap max-w-[280px] z-50 shadow-lg"
          style={{ left: tooltip.x + 12, top: tooltip.y }}>
          {tooltip.text}
        </div>
      )}

      {/* Layer labels */}
      <div className="absolute top-0 left-0 pointer-events-none" style={{ width: MARGIN.left, height: svgHeight }}>
        {layers.map((l) => (
          <div key={`cl-${l.band}`} className="absolute" style={{ top: l.y - 10, left: 8 }}>
            <div className={`text-[10px] font-mono whitespace-nowrap px-1.5 py-0.5 rounded border border-neutral-800/60 ${l.count > 0 ? "text-neutral-400 bg-neutral-900/90 backdrop-blur-sm" : "text-neutral-600 bg-neutral-900/60"}`}>
              Size={l.band}
              <span className="text-neutral-600 ml-1">({l.count})</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
