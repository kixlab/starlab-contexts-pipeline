"use client";

import { useMemo, useState } from "react";
import type { DatasetVideo, Facet } from "@/app/types";
import type { RefinementTree, TreeNode, TreeEdge } from "../utils/treeBuilder";
import { computeMI, computeJointEntropy } from "../utils/treeBuilder";
import {
  facetIdToChipColor,
  computeFacetAbbreviations,
  computeFacetDistributions,
  type FacetValueDist,
} from "@/app/app_analysis/utils/metricsUtils";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type InspectionPanelProps = {
  tree: RefinementTree;
  selectedNodeIds: string[];
  selectedEdgeKey: string | null;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  qualityMetric: "D" | "eta2";
  validPiecesCount: number;
  onNodeClick: (nodeId: string) => void;
  onClose: () => void;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function InspectionPanel({
  tree,
  selectedNodeIds,
  selectedEdgeKey,
  dataset,
  infoTypes,
  qualityMetric,
  validPiecesCount,
  onNodeClick,
  onClose,
}: InspectionPanelProps) {
  const nodeMap = useMemo(() => new Map(tree.nodes.map((n) => [n.id, n])), [tree.nodes]);

  // Determine mode
  const selNodes = selectedNodeIds.map((id) => nodeMap.get(id)).filter(Boolean) as TreeNode[];
  const selEdge = selectedEdgeKey
    ? tree.edges.find((e) => `${e.sourceId}->${e.targetId}` === selectedEdgeKey) ?? null
    : null;

  const isCompare = selNodes.length === 2;
  const isSingle = selNodes.length === 1 && !selEdge;
  const isEdge = selEdge !== null;
  const isEmpty = !isSingle && !isCompare && !isEdge;

  const width = isCompare ? "w-[480px]" : "w-96";

  if (isEmpty) {
    return (
      <div className="w-80 flex-shrink-0 border-l border-neutral-800 bg-neutral-950 flex items-center justify-center">
        <p className="text-neutral-600 text-sm text-center px-6">
          Click a node to inspect it.<br />
          Shift+click two nodes to compare.
        </p>
      </div>
    );
  }

  return (
    <div className={`${width} flex-shrink-0 border-l border-neutral-800 bg-neutral-950 overflow-y-auto`}>
      {/* Close button */}
      <div className="sticky top-0 z-10 flex justify-end p-2 bg-neutral-950/90 backdrop-blur-sm">
        <button onClick={onClose} className="text-neutral-500 hover:text-white text-xs px-2 py-1 rounded hover:bg-neutral-800">
          ✕ Close
        </button>
      </div>

      <div className="px-4 pb-6 space-y-4">
        {isSingle && (
          <SingleNodeView
            node={selNodes[0]}
            tree={tree}
            dataset={dataset}
            infoTypes={infoTypes}
            qualityMetric={qualityMetric}
            onNodeClick={onNodeClick}
          />
        )}
        {isCompare && (
          <ComparisonView
            nodeA={selNodes[0]}
            nodeB={selNodes[1]}
            tree={tree}
            dataset={dataset}
            infoTypes={infoTypes}
            qualityMetric={qualityMetric}
            validPiecesCount={validPiecesCount}
          />
        )}
        {isEdge && (
          <EdgeView
            edge={selEdge!}
            nodeMap={nodeMap}
            qualityMetric={qualityMetric}
            onNodeClick={onNodeClick}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single Node View
// ---------------------------------------------------------------------------

function SingleNodeView({
  node,
  tree,
  dataset,
  infoTypes,
  qualityMetric,
  onNodeClick,
}: {
  node: TreeNode;
  tree: RefinementTree;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  qualityMetric: "D" | "eta2";
  onNodeClick: (id: string) => void;
}) {
  const abbrevs = computeFacetAbbreviations(node.facetSubset);
  const isRoot = tree.roots.includes(node.id);

  const inEdges = useMemo(() => tree.edges.filter((e) => e.targetId === node.id), [tree.edges, node.id]);
  const outEdges = useMemo(() => tree.edges.filter((e) => e.sourceId === node.id), [tree.edges, node.id]);

  const dists = useMemo(
    () => computeFacetDistributions(dataset, node.facetSubset, infoTypes),
    [dataset, node.facetSubset, infoTypes],
  );

  return (
    <>
      {/* Schema */}
      <Section title="Schema">
        <div className="flex flex-wrap gap-1.5">
          {node.facetSubset.map((f) => {
            const c = facetIdToChipColor(f.id);
            return (
              <span key={f.id} className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded"
                style={{ backgroundColor: c + "20", color: c, border: `1px solid ${c}40` }}>
                {abbrevs.get(f.id) ?? f.title.slice(0, 3)}
                <span className="text-[10px] opacity-60">({f.vocabulary.length})</span>
              </span>
            );
          })}
        </div>
        {isRoot && <span className="text-blue-400 text-[10px] font-medium mt-1 block">ROOT</span>}
      </Section>

      {/* Metrics */}
      <Section title="Metrics">
        <MetricGrid>
          <MetricCell label="C" value={node.size} />
          <MetricCell label="D" value={node.D.toFixed(3)} highlight={qualityMetric === "D"} />
          <MetricCell label="η²" value={node.eta2.toFixed(3)} highlight={qualityMetric === "eta2"} />
          <MetricCell label="Stab" value={node.stability.toFixed(3)} />
          <MetricCell label="Cells" value={node.cellCount} />
          <MetricCell label="H" value={node.H.toFixed(3)} />
        </MetricGrid>
      </Section>

      {/* Parents */}
      {inEdges.length > 0 && (
        <Section title={`Parents (${inEdges.length})`}>
          {inEdges.map((e) => (
            <EdgeLink key={e.sourceId} nodeId={e.sourceId} edge={e} direction="parent" onClick={onNodeClick} />
          ))}
        </Section>
      )}

      {/* Children */}
      {outEdges.length > 0 && (
        <Section title={`Children (${outEdges.length})`}>
          {outEdges.map((e) => (
            <EdgeLink key={e.targetId} nodeId={e.targetId} edge={e} direction="child" onClick={onNodeClick} />
          ))}
        </Section>
      )}

      {/* Distributions */}
      {dists.length > 0 && (
        <Section title="Value Distributions">
          {dists.map((d) => (
            <FacetDistBar key={d.facet.id} dist={d} />
          ))}
        </Section>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Comparison View
// ---------------------------------------------------------------------------

function ComparisonView({
  nodeA,
  nodeB,
  tree,
  dataset,
  infoTypes,
  qualityMetric,
  validPiecesCount,
}: {
  nodeA: TreeNode;
  nodeB: TreeNode;
  tree: RefinementTree;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  qualityMetric: "D" | "eta2";
  validPiecesCount: number;
}) {
  const N = validPiecesCount;
  const mi = useMemo(() => computeMI(nodeA, nodeB, N), [nodeA, nodeB, N]);
  const jointH = useMemo(() => computeJointEntropy(nodeA, nodeB, N), [nodeA, nodeB, N]);

  const aIds = new Set(nodeA.facetSubset.map((f) => f.id));
  const bIds = new Set(nodeB.facetSubset.map((f) => f.id));
  const shared = nodeA.facetSubset.filter((f) => bIds.has(f.id));
  const onlyA = nodeA.facetSubset.filter((f) => !bIds.has(f.id));
  const onlyB = nodeB.facetSubset.filter((f) => !aIds.has(f.id));

  const distsA = useMemo(() => computeFacetDistributions(dataset, nodeA.facetSubset, infoTypes), [dataset, nodeA, infoTypes]);
  const distsB = useMemo(() => computeFacetDistributions(dataset, nodeB.facetSubset, infoTypes), [dataset, nodeB, infoTypes]);

  return (
    <>
      {/* Side-by-side schemas */}
      <Section title="Schemas">
        <div className="grid grid-cols-2 gap-2">
          <SchemaCard node={nodeA} label="#1" color="#f59e0b" />
          <SchemaCard node={nodeB} label="#2" color="#22d3ee" />
        </div>
      </Section>

      {/* Facet overlap */}
      <Section title="Facet Overlap">
        {shared.length > 0 && <FacetList label="Shared" facets={shared} />}
        {onlyA.length > 0 && <FacetList label="Only #1" facets={onlyA} />}
        {onlyB.length > 0 && <FacetList label="Only #2" facets={onlyB} />}
      </Section>

      {/* Metrics comparison */}
      <Section title="Metrics">
        <div className="grid grid-cols-4 gap-x-2 gap-y-1 text-xs">
          <span className="text-neutral-500">Metric</span>
          <span className="text-amber-400 text-right">#1</span>
          <span className="text-cyan-400 text-right">#2</span>
          <span className="text-neutral-400 text-right">Δ</span>
          <CompRow label="C" a={nodeA.size} b={nodeB.size} />
          <CompRow label="D" a={nodeA.D} b={nodeB.D} fmt={3} lower />
          <CompRow label="η²" a={nodeA.eta2} b={nodeB.eta2} fmt={3} />
          <CompRow label="Stab" a={nodeA.stability} b={nodeB.stability} fmt={3} />
        </div>
      </Section>

      {/* MI */}
      <Section title="Mutual Information">
        <MetricGrid>
          <MetricCell label="MI" value={mi.toFixed(3)} />
          <MetricCell label="MI/H(#1)" value={nodeA.H > 0 ? (mi / nodeA.H).toFixed(3) : "—"} />
          <MetricCell label="MI/H(#2)" value={nodeB.H > 0 ? (mi / nodeB.H).toFixed(3) : "—"} />
          <MetricCell label="H(joint)" value={jointH.toFixed(3)} />
        </MetricGrid>
      </Section>

      {/* Side-by-side distributions */}
      <Section title="Distributions">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <span className="text-amber-400 text-[10px] font-medium">#1</span>
            {distsA.map((d) => <FacetDistBar key={d.facet.id} dist={d} compact />)}
          </div>
          <div className="space-y-2">
            <span className="text-cyan-400 text-[10px] font-medium">#2</span>
            {distsB.map((d) => <FacetDistBar key={d.facet.id} dist={d} compact />)}
          </div>
        </div>
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Edge View
// ---------------------------------------------------------------------------

function EdgeView({
  edge,
  nodeMap,
  qualityMetric,
  onNodeClick,
}: {
  edge: TreeEdge;
  nodeMap: Map<string, TreeNode>;
  qualityMetric: "D" | "eta2";
  onNodeClick: (id: string) => void;
}) {
  const src = nodeMap.get(edge.sourceId);
  const tgt = nodeMap.get(edge.targetId);
  if (!src || !tgt) return null;

  return (
    <>
      <Section title="Edge">
        <div className="flex items-center gap-2 text-xs">
          <button onClick={() => onNodeClick(edge.sourceId)} className="text-blue-400 hover:underline">{edge.sourceId}</button>
          <span className="text-neutral-500">→</span>
          <button onClick={() => onNodeClick(edge.targetId)} className="text-blue-400 hover:underline">{edge.targetId}</button>
          <EdgeTypeBadge type={edge.edgeType} />
        </div>
      </Section>

      {/* Facet changes */}
      <Section title="Changes">
        {edge.addedFacets.length > 0 && (
          <div className="text-xs">
            <span className="text-green-400">+ Added:</span>{" "}
            {edge.addedFacets.map((f) => f.title).join(", ")}
          </div>
        )}
        {edge.removedFacets.length > 0 && (
          <div className="text-xs">
            <span className="text-red-400">- Removed:</span>{" "}
            {edge.removedFacets.map((f) => f.title).join(", ")}
          </div>
        )}
        {edge.addedFacets.length === 0 && edge.removedFacets.length === 0 && (
          <div className="text-xs text-neutral-500">No facet changes</div>
        )}
      </Section>

      {/* Edge metrics */}
      <Section title="Edge Metrics">
        <MetricGrid>
          <MetricCell label="MI" value={edge.MI.toFixed(3)} />
          <MetricCell label="Refinement" value={edge.refinementScore.toFixed(3)} />
          <MetricCell label="ΔD" value={edge.marginalDGain.toFixed(3)} />
          <MetricCell label="Δη²" value={edge.marginalEta2Gain.toFixed(3)} />
        </MetricGrid>
      </Section>

      {/* Before / after */}
      <Section title="Before / After">
        <div className="grid grid-cols-2 gap-2">
          <SchemaCard node={src} label="Parent" color="#60a5fa" />
          <SchemaCard node={tgt} label="Child" color="#60a5fa" />
        </div>
        <div className="grid grid-cols-4 gap-x-2 gap-y-1 text-xs mt-3">
          <span className="text-neutral-500">Metric</span>
          <span className="text-neutral-300 text-right">Parent</span>
          <span className="text-neutral-300 text-right">Child</span>
          <span className="text-neutral-400 text-right">Δ</span>
          <CompRow label="C" a={src.size} b={tgt.size} />
          <CompRow label="D" a={src.D} b={tgt.D} fmt={3} lower />
          <CompRow label="η²" a={src.eta2} b={tgt.eta2} fmt={3} />
          <CompRow label="Stab" a={src.stability} b={tgt.stability} fmt={3} />
        </div>
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider mb-1.5">{title}</h3>
      {children}
    </div>
  );
}

function MetricGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-3 gap-2">{children}</div>;
}

function MetricCell({ label, value, highlight }: { label: string; value: string | number; highlight?: boolean }) {
  return (
    <div className={`rounded px-2 py-1 ${highlight ? "bg-blue-900/30 border border-blue-700/40" : "bg-neutral-900 border border-neutral-800"}`}>
      <div className="text-[9px] text-neutral-500">{label}</div>
      <div className="text-xs font-mono text-neutral-200">{value}</div>
    </div>
  );
}

function CompRow({ label, a, b, fmt, lower }: { label: string; a: number; b: number; fmt?: number; lower?: boolean }) {
  const fa = fmt != null ? a.toFixed(fmt) : a;
  const fb = fmt != null ? b.toFixed(fmt) : b;
  const delta = b - a;
  const fd = fmt != null ? delta.toFixed(fmt) : delta;
  const better = lower ? delta < -0.001 : delta > 0.001;
  const worse = lower ? delta > 0.001 : delta < -0.001;
  return (
    <>
      <span className="text-neutral-400">{label}</span>
      <span className="text-right font-mono text-neutral-200">{fa}</span>
      <span className="text-right font-mono text-neutral-200">{fb}</span>
      <span className={`text-right font-mono ${better ? "text-green-400" : worse ? "text-red-400" : "text-neutral-500"}`}>
        {delta > 0 ? "+" : ""}{fd}
      </span>
    </>
  );
}

function SchemaCard({ node, label, color }: { node: TreeNode; label: string; color: string }) {
  const abbrevs = computeFacetAbbreviations(node.facetSubset);
  return (
    <div className="rounded border border-neutral-800 p-2 bg-neutral-900">
      <div className="text-[10px] font-medium mb-1" style={{ color }}>{label} · C={node.size}</div>
      <div className="flex flex-wrap gap-1">
        {node.facetSubset.map((f) => {
          const c = facetIdToChipColor(f.id);
          return (
            <span key={f.id} className="text-[9px] px-1.5 py-0.5 rounded"
              style={{ backgroundColor: c + "20", color: c }}>
              {abbrevs.get(f.id) ?? f.title.slice(0, 3)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function EdgeLink({
  nodeId,
  edge,
  direction,
  onClick,
}: {
  nodeId: string;
  edge: TreeEdge;
  direction: "parent" | "child";
  onClick: (id: string) => void;
}) {
  return (
    <button
      onClick={() => onClick(nodeId)}
      className="w-full text-left text-xs px-2 py-1 rounded hover:bg-neutral-800 flex items-center gap-2"
    >
      <span className="text-blue-400 truncate flex-1">{nodeId}</span>
      <EdgeTypeBadge type={edge.edgeType} />
      <span className="text-neutral-500 text-[10px]">ref={edge.refinementScore.toFixed(2)}</span>
    </button>
  );
}

function EdgeTypeBadge({ type }: { type: string }) {
  const color = type === "add" ? "text-green-400 border-green-700/40 bg-green-900/20" : "text-yellow-400 border-yellow-700/40 bg-yellow-900/20";
  return <span className={`text-[9px] px-1 py-0 rounded border ${color}`}>{type}</span>;
}

function FacetDistBar({ dist, compact }: { dist: FacetValueDist; compact?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const c = facetIdToChipColor(dist.facet.id);
  const sorted = [...dist.valueCounts].sort((a, b) => b.count - a.count);
  const shown = expanded ? sorted : sorted.slice(0, compact ? 3 : 5);

  return (
    <div className="text-xs">
      <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-1 text-neutral-300 hover:text-white mb-0.5">
        <span className="text-[10px]" style={{ color: c }}>{dist.facet.title}</span>
        {sorted.length > (compact ? 3 : 5) && (
          <span className="text-neutral-600 text-[9px]">{expanded ? "▾" : `▸ ${sorted.length}`}</span>
        )}
      </button>
      <div className="space-y-0.5">
        {shown.map((v) => (
          <div key={v.label} className="flex items-center gap-1">
            <span className="text-neutral-500 w-16 truncate text-[10px]">{v.label}</span>
            <div className="flex-1 h-2 bg-neutral-800 rounded overflow-hidden">
              <div className="h-full rounded" style={{ width: `${v.percentage}%`, backgroundColor: c + "80" }} />
            </div>
            <span className="text-neutral-600 text-[9px] w-8 text-right">{v.percentage.toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FacetList({ label, facets }: { label: string; facets: Facet[] }) {
  return (
    <div className="text-xs mb-1">
      <span className="text-neutral-500">{label}: </span>
      {facets.map((f) => {
        const c = facetIdToChipColor(f.id);
        return (
          <span key={f.id} className="inline-block text-[10px] px-1.5 py-0 rounded mr-1"
            style={{ backgroundColor: c + "20", color: c }}>
            {f.title}
          </span>
        );
      })}
    </div>
  );
}
