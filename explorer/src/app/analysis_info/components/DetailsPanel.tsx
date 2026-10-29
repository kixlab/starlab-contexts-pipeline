import { useState } from "react";
import type { DatasetVideo, TranscriptEntry, Facet } from "@/app/types";
import { VideoPlayer } from "./VideoPlayer";

export function DetailsPanel({
  pieces,
  dataset,
  facets,
  selectedFacets,
  onClear,
  infoTypes,
}: {
  pieces: DatasetVideo["pieces"];
  dataset: DatasetVideo[];
  facets: Facet[];
  selectedFacets: Record<string, { label: string; title: string }>;
  onClear: () => void;
  infoTypes: readonly string[];
}) {
  const [expandedPieces, setExpandedPieces] = useState<Set<string>>(new Set());
  const facetKeyToTitle = new Map<string, string>(facets.map((f) => [f.id, f.title]));

  // Find the video URL for a given piece
  const getVideoUrlForPiece = (piece: DatasetVideo["pieces"][number]): string | null => {
    // Find the video that contains this piece
    const video = dataset.find(v => v.pieces?.some(p => p.piece_id === piece.piece_id));
    return video?.url || null;
  };

  // Find the previous sentence start time for better context
  const getPreviousSentenceStart = (piece: DatasetVideo["pieces"][number]): number => {
    const video = dataset.find(v => v.pieces?.some(p => p.piece_id === piece.piece_id));
    if (!video?.transcript) return piece.start;

    // Find the transcript entry that overlaps with this piece (more flexible matching)
    const transcriptEntry = video.transcript.find((t: TranscriptEntry) => 
      (t.start <= piece.start && t.end >= piece.start) || 
      (t.start <= piece.end && t.end >= piece.end) ||
      (t.start >= piece.start && t.end <= piece.end)
    );
    
    if (!transcriptEntry) {
      // If no exact match, find the closest transcript entry before the piece
      const beforeEntries = video.transcript.filter((t: TranscriptEntry) => t.end <= piece.start);
      if (beforeEntries.length > 0) {
        const lastBefore = beforeEntries[beforeEntries.length - 1];
        return lastBefore.start;
      }
      return piece.start;
    }

    // Find the previous transcript entry
    const currentIndex = video.transcript.indexOf(transcriptEntry);
    const previousEntry = video.transcript[currentIndex - 1];
    
    return previousEntry ? previousEntry.start : piece.start;
  };
  return (
    <section className="rounded-xl border border-neutral-200 dark:border-neutral-800 p-4 min-w-0 min-h-0 h-full flex flex-col">
      {!pieces.length ? (
        <div className="text-neutral-500 text-sm flex-1 flex items-center justify-center">Select an information unit to see details.</div>
      ) : (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold">Details</h3>
            <button className="text-xs underline" onClick={onClear}>Clear</button>
          </div>
          <div className="space-y-2 flex-1 overflow-y-auto pr-1 min-h-0">
            {pieces.filter(p => infoTypes.includes(p.content_type)).map((p, index) => {
              const isExpanded = expandedPieces.has(p.piece_id);
              
              // Show only facet-value pairs for facets currently shown in FacetPanel (i.e., `facets` prop).
              // Labels may contain additional facet ids (e.g., facet_candidates superset); we hide those here.
              const allFacets = facets.flatMap((f) => {
                const k = f.id;
                const vals = ((p.labels || {}) as Record<string, string[]>)[k] || [];
                const clean = vals.map((v) => (v ?? "").trim()).filter(Boolean);
                return clean.map((v) => ({
                  key: k,
                  value: v,
                  displayKey: f.title,
                  isSelected: selectedFacets[k]?.label === v,
                }));
              });

              // Best-effort previous/next sentence using piece_id pattern: piece_<videoIdx>_<pieceIdx>
              let prevText = "";
              let nextText = "";
              const match = /^piece_(\d+)_(\d+)$/.exec(p.piece_id);
              if (match) {
                const vIdx = Number(match[1]);
                const iIdx = Number(match[2]);
                const video = dataset[vIdx];
                const neighbors = video?.pieces ?? [];
                prevText = neighbors[iIdx - 1]?.content ?? "";
                nextText = neighbors[iIdx + 1]?.content ?? "";
              }

              return (
                <div key={p.piece_id} className="rounded-xl border border-neutral-200 dark:border-neutral-800 p-3">
                  {/* Collapsed view - always visible */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30 px-2 py-0.5 rounded-full">
                          {index + 1}
                        </span>
                        <div className="text-sm font-medium line-clamp-2 flex-1">{p.content}</div>
                      </div>
                      <div className="text-xs text-neutral-500 mb-2">{p.content_type} • {p.start}s - {p.end}s</div>
                      {allFacets.length > 0 && (
                        <div className="flex flex-wrap gap-1 text-[10px]">
                          {allFacets.map(({ key, value, displayKey, isSelected }) => (
                            <span 
                              key={`${p.piece_id}-${key}-${value}`} 
                              className={`px-2 py-0.5 rounded-full border transition-all ${
                                isSelected 
                                  ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium' 
                                  : 'bg-neutral-100 dark:bg-neutral-800 border-neutral-200 dark:border-neutral-700 text-neutral-600 dark:text-neutral-300'
                              }`}
                            >
                              {displayKey}: {value}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => {
                        setExpandedPieces(prev => {
                          const newSet = new Set(prev);
                          if (newSet.has(p.piece_id)) {
                            newSet.delete(p.piece_id);
                          } else {
                            newSet.add(p.piece_id);
                          }
                          return newSet;
                        });
                      }}
                      className="ml-2 p-1 text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors"
                      aria-label={isExpanded ? "Collapse details" : "Expand details"}
                    >
                      <svg className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                  </div>

                  {/* Expanded view - only when expanded */}
                  {isExpanded && (
                    <div className="mt-3 pt-3 border-t border-neutral-200 dark:border-neutral-700 bg-blue-50 dark:bg-blue-950/30 rounded-lg p-3 -mx-3 -mb-3">
                      {(() => {
                        const videoUrl = getVideoUrlForPiece(p);
                        const startTime = getPreviousSentenceStart(p);
                        console.log(`Piece: ${p.content.substring(0, 50)}... | Piece start: ${p.start}s | Video start: ${startTime}s`);
                        return videoUrl ? (
                          <VideoPlayer 
                            videoUrl={videoUrl}
                            startTime={startTime}
                            endTime={p.end}
                            className="mb-2"
                          />
                        ) : (
                          <div className="mb-2 w-full h-28 rounded-lg bg-neutral-200 dark:bg-neutral-800 flex items-center justify-center text-xs text-neutral-500">
                            Video not available
                          </div>
                        );
                      })()}
                      {(prevText || p.content || nextText) && (
                        <div className="text-sm leading-6 mb-3">
                          {prevText && <span className="text-neutral-500 italic">{prevText} </span>}
                          <span className="font-semibold">{p.content}</span>
                          {nextText && <span className="text-neutral-500 italic"> {nextText}</span>}
                        </div>
                      )}
                      <div className="flex flex-wrap gap-1 text-[10px]">
                        {allFacets.map(({ key, value, displayKey, isSelected }) => (
                          <span 
                            key={`${p.piece_id}-${key}-${value}`} 
                            className={`px-2 py-0.5 rounded-full border transition-all ${
                              isSelected 
                                ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300 font-medium' 
                                : 'bg-neutral-100 dark:bg-neutral-800 border-neutral-200 dark:border-neutral-700 text-neutral-600 dark:text-neutral-300'
                            }`}
                          >
                            {displayKey}: {value}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

