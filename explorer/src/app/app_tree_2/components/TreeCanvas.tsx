"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TreeNode, TreeEdge, RefinementTree } from "../utils/treeBuilder";
import { NODE_WIDTH, NODE_HEIGHT, MARGIN } from "../utils/stableLayout";
import type { VisibleEdge, NodePosition } from "../types";
import {
  facetIdToChipColor,
  computeFacetAbbreviations,
} from "@/app/app_analysis/utils/metricsUtils";

// ---------------------------------------------------------------------------
// Animated node tracking
// ---------------------------------------------------------------------------

type AnimPhase = "entering" | "stable" | "exiting";
type AnimEntry = { nodeId: string; phase: AnimPhase; x: number; y: number; opacity: number };

function useAnimatedNodes(
  visibleNodes: TreeNode[],
  positions: Map<string, NodePosition>,
): AnimEntry[] {
  const prevIds = useRef(new Set<string>());
  const exitingRef = useRef(new Map<string, { x: number; y: number; timer: ReturnType<typeof setTimeout> }>());
  const [, forceUpdate] = useState(0);

  const currentIds = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);

  // Detect exits
  useEffect(() => {
    for (const id of prevIds.current) {
      if (!currentIds.has(id) && !exitingRef.current.has(id)) {
        // Preserve last known position
        const prev = positions.get(id);
        const x = prev?.x ?? 0;
        const y = prev?.y ?? 0;
        const timer = setTimeout(() => {
          exitingRef.current.delete(id);
          forceUpdate((c) => c + 1);
        }, 300);
        exitingRef.current.set(id, { x, y, timer });
        forceUpdate((c) => c + 1);
      }
    }
    prevIds.current = currentIds;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIds]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const { timer } of exitingRef.current.values()) clearTimeout(timer);
    };
  }, []);

  const entries: AnimEntry[] = [];

  // Exiting nodes
  for (const [id, { x, y }] of exitingRef.current) {
    entries.push({ nodeId: id, phase: "exiting", x, y, opacity: 0 });
  }

  // Visible nodes
  const prevSet = prevIds.current;
  for (const n of visibleNodes) {
    const pos = positions.get(n.id);
    if (!pos || pos.isDummy) continue;
    const isNew = !prevSet.has(n.id) || exitingRef.current.has(n.id);
    // Cancel exit if re-appearing
    if (exitingRef.current.has(n.id)) {
      const ex = exitingRef.current.get(n.id)!;
      clearTimeout(ex.timer);
      exitingRef.current.delete(n.id);
    }
    entries.push({
      nodeId: n.id,
      phase: isNew ? "entering" : "stable",
      x: pos.x,
      y: pos.y,
      opacity: 1,
    });
  }

  return entries;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type TreeCanvasProps = {
  tree: RefinementTree;
  visibleNodes: TreeNode[];
  visibleEdges: VisibleEdge[];
  positions: Map<string, NodePosition>;
  expandedNodeIds: Set<string>;
  childrenCountMap: Map<string, number>;
  qualityMetric: "D" | "eta2";
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
  visibleNodes,
  visibleEdges,
  positions,
  expandedNodeIds,
  childrenCountMap,
  qualityMetric,
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
    for (const pos of positions.values()) {
      maxY = Math.max(maxY, pos.y + NODE_HEIGHT / 2 + MARGIN.bottom);
    }
    return maxY;
  }, [positions]);

  // Layers for background stripes + labels
  const layers = useMemo(() => {
    const bandSet = new Set(visibleNodes.map((n) => n.sizeBand));
    const sorted = Array.from(bandSet).sort((a, b) => a - b);
    return sorted.map((band) => {
      const nodes = visibleNodes.filter((n) => n.sizeBand === band);
      const pos = positions.get(nodes[0]?.id ?? "");
      return { band, count: nodes.length, y: pos?.y ?? 0 };
    });
  }, [visibleNodes, positions]);

  // Row height estimate
  const rowHeight = layers.length > 1
    ? Math.abs((layers[1]?.y ?? 0) - (layers[0]?.y ?? 0))
    : 100;

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

  // Scroll state for sticky labels
  const [scrollLeft, setScrollLeft] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (el) { setScrollLeft(el.scrollLeft); setScrollTop(el.scrollTop); }
  }, []);

  // Animated nodes
  const nodeMap = useMemo(() => new Map(tree.nodes.map((n) => [n.id, n])), [tree.nodes]);
  const animEntries = useAnimatedNodes(visibleNodes, positions);

  // Edge helpers
  const edgePath = useCallback((e: TreeEdge) => {
    const src = positions.get(e.sourceId);
    const tgt = positions.get(e.targetId);
    if (!src || !tgt) return "";
    const sy = src.y + NODE_HEIGHT / 2;
    const ty = tgt.y - NODE_HEIGHT / 2;
    const midY = (sy + ty) / 2;
    return `M ${src.x} ${sy} C ${src.x} ${midY}, ${tgt.x} ${midY}, ${tgt.x} ${ty}`;
  }, [positions]);

  const EDGE_COLORS: Record<string, string> = { add: "#60a5fa", restructure: "#facc15" };

  const handleBgClick = useCallback(() => {
    onEdgeClick(null);
    onDeselectAll();
  }, [onEdgeClick, onDeselectAll]);

  // Metric helpers
  const metricVal = (n: TreeNode) => qualityMetric === "D" ? n.D : n.eta2;
  const metricLabel = (n: TreeNode) => {
    const v = metricVal(n);
    return qualityMetric === "D" ? `D=${v.toFixed(3)}` : `η²=${v.toFixed(3)}`;
  };
  const nodeFill = (n: TreeNode) => {
    const v = metricVal(n);
    const t = qualityMetric === "D" ? 1 - Math.min(1, v / 4) : Math.min(1, v);
    return `rgb(${30 + t * 30},${30 + t * 60},${50 + t * 80})`;
  };

  if (visibleNodes.length === 0) {
    return (
      <div ref={containerRef} className="w-full h-full flex items-center justify-center text-neutral-500 text-sm">
        No schemas pass the current filters. Click Build to generate the tree.
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-auto" onScroll={handleScroll}>
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
        {visibleEdges.map((e) => {
          const key = `${e.sourceId}->${e.targetId}`;
          const isSel = key === selectedEdgeKey;
          const color = EDGE_COLORS[e.edgeType] ?? "#6b7280";
          const dash = e.contracted ? "4 3" : e.edgeType === "restructure" ? "6 3" : "none";
          const op = e.contracted ? 0.4 : 0.7;
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
                      text: `${e.contracted ? "[contracted] " : ""}[${e.edgeType}] MI=${e.MI.toFixed(3)}  ref=${e.refinementScore.toFixed(3)}\nΔD=${e.marginalDGain.toFixed(3)}  Δη²=${e.marginalEta2Gain.toFixed(3)}`,
                    });
                  }
                }}
                onMouseLeave={() => setTooltip(null)}
              />
              <path d={edgePath(e)} fill="none"
                stroke={isSel ? "#f59e0b" : color}
                strokeWidth={isSel ? 2.5 : e.contracted ? 1 : 1 + e.refinementScore * 2}
                strokeOpacity={isSel ? 1 : op}
                strokeDasharray={isSel ? "6 3" : dash}
                markerEnd={`url(#t2-arrow-${isSel ? "sel" : e.edgeType})`}
                className="pointer-events-none"
                style={{ transition: "opacity 200ms ease" }}
              />
            </g>
          );
        })}

        {/* Nodes */}
        {animEntries.map((anim) => {
          const node = nodeMap.get(anim.nodeId);
          if (!node) return null;
          const isRoot = tree.roots.includes(node.id);
          const selIdx = selectedNodeIds.indexOf(node.id);
          const isSel = selIdx >= 0;
          const isHov = node.id === hoveredNodeId;
          const isHi = highlightedNodeIds.has(node.id);
          const isExp = expandedNodeIds.has(node.id);
          const numCh = childrenCountMap.get(node.id) ?? 0;
          const abbrevs = computeFacetAbbreviations(node.facetSubset);
          const selColor = selIdx === 0 ? "#f59e0b" : selIdx === 1 ? "#22d3ee" : null;

          return (
            <g
              key={node.id}
              style={{
                transform: `translate(${anim.x - NODE_WIDTH / 2}px, ${anim.y - NODE_HEIGHT / 2}px)`,
                opacity: anim.opacity,
                transition: "transform 300ms cubic-bezier(0.4,0,0.2,1), opacity 300ms ease",
              }}
              className="cursor-pointer"
              onClick={(ev) => { ev.stopPropagation(); onNodeClick(node.id, ev.shiftKey); }}
              onMouseEnter={() => onNodeHover(node.id)}
              onMouseLeave={() => onNodeHover(null)}
            >
              {/* Highlight ring */}
              {isHi && !isSel && (
                <rect x={-4} y={-4} width={NODE_WIDTH + 8} height={NODE_HEIGHT + 8} rx={11}
                  fill="none" stroke="#60a5fa" strokeWidth={2} strokeDasharray="4 2" strokeOpacity={0.8} />
              )}

              {/* Background */}
              <rect width={NODE_WIDTH} height={NODE_HEIGHT} rx={8} fill={nodeFill(node)}
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

              {/* Facet chips */}
              <g transform="translate(6, 8)">
                {node.facetSubset.map((facet, fi) => {
                  const c = facetIdToChipColor(facet.id);
                  const abbr = abbrevs.get(facet.id) ?? facet.title.slice(0, 3);
                  const cx = fi * 34;
                  if (cx + 30 > NODE_WIDTH - 12) return null;
                  return (
                    <g key={facet.id} transform={`translate(${cx}, 0)`}>
                      <rect width={30} height={14} rx={3} fill={c + "30"} stroke={c + "60"} strokeWidth={0.5} />
                      <text x={15} y={10} textAnchor="middle" className="text-[8px] font-medium" fill={c}>{abbr}</text>
                    </g>
                  );
                })}
              </g>

              {/* Metrics */}
              <text x={NODE_WIDTH / 2} y={35} textAnchor="middle" className="text-[9px]" fill="#a3a3a3">
                C={node.size} · {metricLabel(node)}
              </text>

              {/* Bottom: expand badge or cell stats */}
              {numCh > 0 ? (
                <text x={NODE_WIDTH / 2} y={50} textAnchor="middle" className="text-[8px]"
                  fill={isExp ? "#60a5fa" : "#737373"}>
                  {isExp ? "▼ expanded" : `▶ ${numCh} children`}
                </text>
              ) : (
                <text x={NODE_WIDTH / 2} y={50} textAnchor="middle" className="text-[8px]" fill="#737373">
                  cells={node.cellCount} · stab={node.stability.toFixed(2)}
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

      {/* Layer labels (sticky left gutter) */}
      <div className="absolute top-0 left-0 pointer-events-none z-20" style={{ width: MARGIN.left, height: "100%" }}>
        {layers.map((l) => {
          const labelY = l.y - scrollTop;
          if (labelY < -20 || labelY > 2000) return null;
          return (
            <div key={`cl-${l.band}`} className="absolute" style={{ top: labelY - 10, left: 8 }}>
              <div className="text-[10px] font-mono text-neutral-400 bg-neutral-900/90 backdrop-blur-sm px-1.5 py-0.5 rounded border border-neutral-800/60 whitespace-nowrap">
                C={l.band}
                <span className="text-neutral-600 ml-1">({l.count})</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
