"use client";

import { useMemo, useState } from "react";
import type { DatasetVideo, Facet } from "@/app/types";
import { facetToLabelKey } from "@/app/utils/facetStats";
import type { RefinementTree, TreeNode, TreeEdge } from "../utils/treeBuilder";
import { computeMI, computeEffectiveness } from "../utils/treeBuilder";
import {
  facetIdToChipColor,
  computeFacetDistributions,
  type FacetValueDist,
} from "@/app/app_analysis/utils/metricsUtils";

// ---------------------------------------------------------------------------
// Metric tooltip text
// ---------------------------------------------------------------------------

const METRIC_TOOLTIPS: Record<string, string> = {
  Size: "Number of values in the schema. More values enable finer annotations.",
  Effectiveness: "How precisely the schema identifies content. Higher = viewers find specific content more easily.",
  Refinement: "How much a child preserves its parent's organization. Higher = more consistent structure.",
};

// ---------------------------------------------------------------------------
// Instant tooltip wrapper (no delay, unlike native title)
// ---------------------------------------------------------------------------

function Tip({ title, text, children, as: Tag = "div" }: { title?: string; text: string; children: React.ReactNode; as?: "div" | "span" }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  return (
    <Tag
      className="relative"
      onMouseEnter={(e: React.MouseEvent) => {
        const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
        setPos({ x: rect.left + rect.width / 2, y: rect.top });
      }}
      onMouseLeave={() => setPos(null)}
    >
      {children}
      {pos && (
        <div
          className="fixed z-[100] pointer-events-none px-2.5 py-1.5 rounded bg-neutral-800 border border-neutral-700 text-xs text-neutral-200 shadow-lg max-w-[280px]"
          style={{ left: pos.x, top: pos.y - 4, transform: "translate(-50%, -100%)" }}
        >
          {title && <div className="font-semibold text-white mb-0.5 whitespace-nowrap">{title}</div>}
          <div className="whitespace-pre-wrap">{text}</div>
        </div>
      )}
    </Tag>
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type InspectionPanelProps = {
  tree: RefinementTree;
  selectedNodeIds: string[];
  selectedEdgeKey: string | null;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  numUniqueUnits: number;
  validPiecesCount: number;
  pieceUnitIds: Uint32Array;
  unitTotals: Uint32Array;
  onNodeClick: (nodeId: string, trigger: "child" | "parent") => void;
  onExpandDist?: (nodeId: string, facetId: string) => void;
  onCopyId?: (facetIds: string) => void;
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
  numUniqueUnits,
  validPiecesCount,
  pieceUnitIds,
  unitTotals,
  onNodeClick,
  onExpandDist,
  onCopyId,
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

  if (isEmpty) {
    return (
      <div className="w-96 flex-shrink-0 border-l border-neutral-800 bg-neutral-950 flex items-center justify-center">
        <p className="text-neutral-600 text-sm text-center px-6">
          Click a node to inspect it.<br />
          Shift+click two nodes to compare.
        </p>
      </div>
    );
  }

  return (
    <div className="w-96 flex-shrink-0 border-l border-neutral-800 bg-neutral-950 overflow-y-auto">
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
            numUniqueUnits={numUniqueUnits}
            nodeMap={nodeMap}
            onNodeClick={onNodeClick}
            onExpandDist={onExpandDist}
            onCopyId={onCopyId}
          />
        )}
        {isCompare && (
          <ComparisonView
            nodeA={selNodes[0]}
            nodeB={selNodes[1]}
            dataset={dataset}
            infoTypes={infoTypes}
            numUniqueUnits={numUniqueUnits}
            validPiecesCount={validPiecesCount}
            pieceUnitIds={pieceUnitIds}
            unitTotals={unitTotals}
            onCopyId={onCopyId}
          />
        )}
        {isEdge && (
          <EdgeView
            edge={selEdge!}
            nodeMap={nodeMap}
            numUniqueUnits={numUniqueUnits}
            onNodeClick={onNodeClick}
            onCopyId={onCopyId}
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
  numUniqueUnits,
  nodeMap,
  onNodeClick,
  onExpandDist,
  onCopyId,
}: {
  node: TreeNode;
  tree: RefinementTree;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  numUniqueUnits: number;
  nodeMap: Map<string, TreeNode>;
  onNodeClick: (id: string, trigger: "child" | "parent") => void;
  onExpandDist?: (nodeId: string, facetId: string) => void;
  onCopyId?: (facetIds: string) => void;
}) {
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
      <Section title="Schema" extra={<CopyFacetIdsButton facets={node.facetSubset} onCopy={onCopyId} />}>
        <div className="flex flex-wrap gap-1.5">
          {node.facetSubset.map((f) => {
            const c = facetIdToChipColor(f.id);
            return (
              <Tip key={f.id} title={f.title} text={f.definition} as="span">
                <span className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded"
                  style={{ backgroundColor: c + "20", color: c, border: `1px solid ${c}40` }}>
                  {f.title}
                  <span className="text-[11px] opacity-60">({f.vocabulary.length})</span>
                </span>
              </Tip>
            );
          })}
        </div>
        {isRoot && <span className="text-blue-400 text-xs font-medium mt-1 block">ROOT</span>}
      </Section>

      {/* Metrics */}
      <Section title="Metrics">
        <MetricGrid>
          <MetricCell label="Size" value={node.size} tooltip={METRIC_TOOLTIPS.Size} />
          <MetricCell label="Effectiveness" value={`${computeEffectiveness(node.D, numUniqueUnits).toFixed(0)}%`} tooltip={METRIC_TOOLTIPS.Effectiveness} />
        </MetricGrid>
      </Section>

      {/* Parents */}
      {inEdges.length > 0 && (
        <Section title={`Parents (${inEdges.length})`}>
          {inEdges.map((e) => (
            <EdgeLink key={e.sourceId} nodeId={e.sourceId} edge={e} nodeMap={nodeMap} onClick={(id) => onNodeClick(id, "parent")} />
          ))}
        </Section>
      )}

      {/* Children */}
      {outEdges.length > 0 && (
        <Section title={`Children (${outEdges.length})`}>
          {outEdges.map((e) => (
            <EdgeLink key={e.targetId} nodeId={e.targetId} edge={e} nodeMap={nodeMap} onClick={(id) => onNodeClick(id, "child")} />
          ))}
        </Section>
      )}

      {/* Distributions */}
      {dists.length > 0 && (
        <Section title="Definitions & Distributions" extra={<span className="text-[9px] text-neutral-600 font-normal normal-case tracking-normal">click facets to expand</span>}>
          {dists.map((d) => (
            <FacetDistBar key={d.facet.id} dist={d} dataset={dataset} infoTypes={infoTypes}
              onExpand={onExpandDist ? () => onExpandDist(node.id, d.facet.id) : undefined} />
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
  dataset,
  infoTypes,
  numUniqueUnits,
  validPiecesCount,
  pieceUnitIds,
  unitTotals,
  onCopyId,
}: {
  nodeA: TreeNode;
  nodeB: TreeNode;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  numUniqueUnits: number;
  validPiecesCount: number;
  pieceUnitIds: Uint32Array;
  unitTotals: Uint32Array;
  onCopyId?: (facetIds: string) => void;
}) {
  const mi = useMemo(
    () => computeMI(nodeA, nodeB, pieceUnitIds, unitTotals, numUniqueUnits),
    [nodeA, nodeB, pieceUnitIds, unitTotals, numUniqueUnits],
  );

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
          <SchemaCard node={nodeA} label="#1" color="#f59e0b" onCopyId={onCopyId} />
          <SchemaCard node={nodeB} label="#2" color="#22d3ee" onCopyId={onCopyId} />
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
          <CompRow label="Size" a={nodeA.size} b={nodeB.size} tooltip={METRIC_TOOLTIPS.Size} />
          <CompRow label="Effectiveness" a={computeEffectiveness(nodeA.D, numUniqueUnits)} b={computeEffectiveness(nodeB.D, numUniqueUnits)} fmt={0} suffix="%" tooltip={METRIC_TOOLTIPS.Effectiveness} />
        </div>
      </Section>

      {/* Overlap */}
      <Section title="Overlap">
        <MetricGrid>
          <MetricCell label="Overlap with #1" value={nodeA.H > 0 ? `${Math.round((mi / nodeA.H) * 100)}%` : "—"} />
          <MetricCell label="Overlap with #2" value={nodeB.H > 0 ? `${Math.round((mi / nodeB.H) * 100)}%` : "—"} />
        </MetricGrid>
      </Section>

      {/* Side-by-side distributions */}
      <Section title="Distributions">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <span className="text-amber-400 text-xs font-medium">#1</span>
            {distsA.map((d) => <FacetDistBar key={d.facet.id} dist={d} dataset={dataset} infoTypes={infoTypes} />)}
          </div>
          <div className="space-y-2">
            <span className="text-cyan-400 text-xs font-medium">#2</span>
            {distsB.map((d) => <FacetDistBar key={d.facet.id} dist={d} dataset={dataset} infoTypes={infoTypes} />)}
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
  numUniqueUnits,
  onNodeClick,
  onCopyId,
}: {
  edge: TreeEdge;
  nodeMap: Map<string, TreeNode>;
  numUniqueUnits: number;
  onNodeClick: (id: string, trigger: "child" | "parent") => void;
  onCopyId?: (facetIds: string) => void;
}) {
  const src = nodeMap.get(edge.sourceId);
  const tgt = nodeMap.get(edge.targetId);
  if (!src || !tgt) return null;

  return (
    <>
      <Section title="Edge">
        <div className="flex items-center gap-2 text-xs">
          <button onClick={() => onNodeClick(edge.sourceId, "parent")} className="text-blue-400 hover:underline">{nodeLabel(src)}</button>
          <span className="text-neutral-500">→</span>
          <button onClick={() => onNodeClick(edge.targetId, "child")} className="text-blue-400 hover:underline">{nodeLabel(tgt)}</button>
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
          <MetricCell label="Refinement" value={`${Math.round(edge.refinementScore * 100)}%`} tooltip={METRIC_TOOLTIPS.Refinement} />
        </MetricGrid>
      </Section>

      {/* Before / after */}
      <Section title="Before / After">
        <div className="grid grid-cols-2 gap-2">
          <SchemaCard node={src} label="Parent" color="#60a5fa" onCopyId={onCopyId} />
          <SchemaCard node={tgt} label="Child" color="#60a5fa" onCopyId={onCopyId} />
        </div>
        <div className="grid grid-cols-4 gap-x-2 gap-y-1 text-xs mt-3">
          <span className="text-neutral-500">Metric</span>
          <span className="text-neutral-300 text-right">Parent</span>
          <span className="text-neutral-300 text-right">Child</span>
          <span className="text-neutral-400 text-right">Δ</span>
          <CompRow label="Size" a={src.size} b={tgt.size} tooltip={METRIC_TOOLTIPS.Size} />
          <CompRow label="Effectiveness" a={computeEffectiveness(src.D, numUniqueUnits)} b={computeEffectiveness(tgt.D, numUniqueUnits)} fmt={0} suffix="%" tooltip={METRIC_TOOLTIPS.Effectiveness} />
        </div>
      </Section>
    </>
  );
}

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

function nodeLabel(node: TreeNode): string {
  return node.facetSubset.map((f) => f.title).join(" + ");
}

function CopyFacetIdsButton({ facets, onCopy }: { facets: Facet[]; onCopy?: (facetIds: string) => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        const ids = facets.map((f) => f.id).join(", ");
        const write = navigator.clipboard?.writeText
          ? navigator.clipboard.writeText(ids)
          : new Promise<void>((resolve) => {
              const ta = document.createElement("textarea");
              ta.value = ids;
              ta.style.position = "fixed";
              ta.style.opacity = "0";
              document.body.appendChild(ta);
              ta.select();
              document.execCommand("copy");
              document.body.removeChild(ta);
              resolve();
            });
        write.then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
          if (onCopy) onCopy(ids);
        });
      }}
      className="text-[9px] px-1.5 py-0.5 rounded border border-neutral-700 text-neutral-400 hover:text-white hover:border-neutral-500 transition-colors"
    >
      {copied ? "Copied!" : "Copy IDs"}
    </button>
  );
}

function Section({ title, extra, children }: { title: string; extra?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-1.5">
        <h3 className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">{title}</h3>
        {extra}
      </div>
      {children}
    </div>
  );
}

function MetricGrid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-3 gap-2">{children}</div>;
}

function MetricCell({ label, value, highlight, tooltip }: { label: string; value: string | number; highlight?: boolean; tooltip?: string }) {
  const inner = (
    <div className={`rounded px-2 py-1 ${highlight ? "bg-blue-900/30 border border-blue-700/40" : "bg-neutral-900 border border-neutral-800"}`}>
      <div className="text-[9px] text-neutral-500">{label}</div>
      <div className="text-xs font-mono text-neutral-200">{value}</div>
    </div>
  );
  return tooltip ? <Tip text={tooltip}>{inner}</Tip> : inner;
}

function CompRow({ label, a, b, fmt, lower, suffix, tooltip }: { label: string; a: number; b: number; fmt?: number; lower?: boolean; suffix?: string; tooltip?: string }) {
  const s = suffix ?? "";
  const fa = fmt != null ? a.toFixed(fmt) : a;
  const fb = fmt != null ? b.toFixed(fmt) : b;
  const delta = b - a;
  const fd = fmt != null ? delta.toFixed(fmt) : delta;
  const better = lower ? delta < -0.001 : delta > 0.001;
  const worse = lower ? delta > 0.001 : delta < -0.001;
  return (
    <>
      {tooltip ? (
        <Tip text={tooltip} as="span"><span className="text-neutral-400">{label}</span></Tip>
      ) : (
        <span className="text-neutral-400">{label}</span>
      )}
      <span className="text-right font-mono text-neutral-200">{fa}{s}</span>
      <span className="text-right font-mono text-neutral-200">{fb}{s}</span>
      <span className={`text-right font-mono ${better ? "text-green-400" : worse ? "text-red-400" : "text-neutral-500"}`}>
        {delta > 0 ? "+" : ""}{fd}{s}
      </span>
    </>
  );
}

function SchemaCard({ node, label, color, onCopyId }: { node: TreeNode; label: string; color: string; onCopyId?: (facetIds: string) => void }) {
  return (
    <div className="rounded border border-neutral-800 p-2 bg-neutral-900">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-xs font-medium" style={{ color }}>{label} · Size={node.size}</span>
        <CopyFacetIdsButton facets={node.facetSubset} onCopy={onCopyId} />
      </div>
      <div className="flex flex-wrap gap-1">
        {node.facetSubset.map((f) => {
          const c = facetIdToChipColor(f.id);
          return (
            <span key={f.id} className="text-[9px] px-1.5 py-0.5 rounded"
              title={f.id}
              style={{ backgroundColor: c + "20", color: c }}>
              {f.title}
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
  nodeMap,
  onClick,
}: {
  nodeId: string;
  edge: TreeEdge;
  nodeMap: Map<string, TreeNode>;
  onClick: (id: string) => void;
}) {
  const target = nodeMap.get(nodeId);
  const label = target ? nodeLabel(target) : nodeId;
  return (
    <button
      onClick={() => onClick(nodeId)}
      className="w-full text-left text-xs px-2 py-1 rounded hover:bg-neutral-800 flex items-center gap-2"
    >
      <span className="text-blue-400 truncate flex-1">{label}</span>
      <EdgeTypeBadge type={edge.edgeType} />
      <span className="text-neutral-500 text-[11px]">Ref={Math.round(edge.refinementScore * 100)}%</span>
    </button>
  );
}

function EdgeTypeBadge({ type }: { type: string }) {
  const color = type === "add" ? "text-green-400 border-green-700/40 bg-green-900/20" : "text-yellow-400 border-yellow-700/40 bg-yellow-900/20";
  return <span className={`text-[9px] px-1 py-0 rounded border ${color}`}>{type}</span>;
}

function FacetDistBar({
  dist,
  dataset,
  infoTypes,
  onExpand,
}: {
  dist: FacetValueDist;
  dataset: DatasetVideo[];
  infoTypes: string[] | null;
  onExpand?: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const c = facetIdToChipColor(dist.facet.id);
  const sorted = [...dist.valueCounts].sort((a, b) => b.count - a.count);

  // Vocabulary lookup: label → definition
  const vocabMap = useMemo(() => {
    const m = new Map<string, string>();
    for (const v of dist.facet.vocabulary) m.set(v.label, v.definition);
    return m;
  }, [dist.facet.vocabulary]);

  // Compute up to 2 example contents per value (only when details shown)
  const examples = useMemo(() => {
    if (!showDetails) return new Map<string, string[]>();
    const key = facetToLabelKey(dist.facet);
    const rawPieces = dataset.flatMap((v) => v.pieces ?? []);
    const pieces = infoTypes?.length
      ? rawPieces.filter((p) => infoTypes.includes(p.content_type ?? ""))
      : rawPieces;
    const result = new Map<string, string[]>();
    for (const piece of pieces) {
      const vals = (piece.labels?.[key] as string[] | undefined) ?? [];
      for (const val of vals) {
        const arr = result.get(val);
        if (!arr) {
          result.set(val, [piece.content]);
        } else if (arr.length < 2) {
          arr.push(piece.content);
        }
      }
    }
    return result;
  }, [showDetails, dist.facet, dataset, infoTypes]);

  return (
    <div className="text-xs">
      {/* Facet title — click to toggle details */}
      <button
        onClick={() => {
          const next = !showDetails;
          setShowDetails(next);
          if (next && onExpand) onExpand();
        }}
        className="flex items-center gap-1.5 text-neutral-300 hover:text-white mb-0.5 group"
      >
        <span className="text-neutral-500 text-[10px] group-hover:text-neutral-300 transition-colors">{showDetails ? "▾" : "▸"}</span>
        <span className="text-xs font-semibold group-hover:underline" style={{ color: c }}>{dist.facet.title}</span>
      </button>

      {/* Expanded facet details: definition + guidelines */}
      {showDetails && (
        <div className="ml-3 mb-1.5 space-y-1 text-xs text-neutral-300 border-l border-neutral-800 pl-2">
          <p>{dist.facet.definition}</p>
          {dist.facet.guidelines && dist.facet.guidelines.length > 0 && (
            <ul className="list-disc list-inside space-y-0.5 text-neutral-500">
              {dist.facet.guidelines.map((g, i) => (
                <li key={i}>{g}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* All values — always shown */}
      <div className="space-y-0.5">
        {sorted.map((v) => (
          <div key={v.label}>
            <div className="flex items-center gap-1">
              <span className="text-neutral-400 w-20 truncate text-xs" title={v.label}>{v.label}</span>
              <div className="flex-1 h-2 bg-neutral-800 rounded overflow-hidden">
                <div className="h-full rounded" style={{ width: `${v.percentage}%`, backgroundColor: c + "80" }} />
              </div>
              <span className="text-neutral-600 text-[9px] w-8 text-right">{v.percentage.toFixed(0)}%</span>
            </div>
            {/* Expanded value details */}
            {showDetails && (
              <div className="ml-3 mb-1 text-[11px] text-neutral-400 border-l border-neutral-800 pl-2 space-y-0.5">
                <div><span className="text-neutral-600">Name:</span> {v.label}</div>
                {vocabMap.has(v.label) && (
                  <div><span className="text-neutral-600">Def:</span> {vocabMap.get(v.label)}</div>
                )}
                {(examples.get(v.label) ?? []).map((ex, i) => (
                  <div key={i} className="text-neutral-600 italic truncate" title={ex}>
                    e.g. &ldquo;{ex}&rdquo;
                  </div>
                ))}
              </div>
            )}
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
          <span key={f.id} className="inline-block text-xs px-1.5 py-0.5 rounded mr-1"
            style={{ backgroundColor: c + "20", color: c }}>
            {f.title}
          </span>
        );
      })}
    </div>
  );
}
