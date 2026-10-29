"use client";

import { useState } from "react";
import { ProgressBar } from "./ProgressBar";
import { Tooltip, type TooltipData } from "@/app/components/Tooltip";
import { ALL_TYPES_DEFINITION, INFO_TYPE_DEFINITIONS, PER_TYPE_COVERAGE_DEFINITION } from "@/app/constants/infoTypeDefinitions";
import { getInfoTypeColor, shouldUseWhiteText } from "@/app/constants/colors";

type InfoTypeWithAll = string; // either "All" or a full `content_type` value from dataset configuration
type SetStateAction<T> = T | ((prev: T) => T);

export function StatsPanel({
  totalVideos,
  videoCount,
  selectedFacets,
  coverageByType,
  selectedType,
  setSelectedType,
  infoTypes,
  availableInfoTypes,
  setInfoTypes,
}: {
  totalVideos: number;
  videoCount: number;
  selectedFacets: Record<string, { label: string; title: string }>;
  coverageByType: { totalCoverage: number; byType: Record<string, number> };
  selectedType: InfoTypeWithAll;
  setSelectedType: (t: InfoTypeWithAll) => void;
  infoTypes: readonly string[];
  availableInfoTypes: readonly string[];
  setInfoTypes: (next: SetStateAction<string[]>) => void;
}) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [addQuery, setAddQuery] = useState("");

  const handleMouseEnter = (e: React.MouseEvent, title: string, text: string) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltip({
      x: rect.left,
      y: rect.top,
      title,
      text
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  const selectedSet = new Set(infoTypes);
  const addableTypes = (availableInfoTypes || [])
    .filter((t) => !selectedSet.has(t))
    .filter((t) => {
      if (!addQuery.trim()) return true;
      return t.toLowerCase().includes(addQuery.trim().toLowerCase());
    });

  const getInfoTypeDefinition = (label: string) => {
    // We store definitions keyed by the coarse category (e.g., "Method") but
    // content_type strings may look like "Method - Instruction".
    const coarse = label.split(" - ")[0]?.trim() || label;
    return (
      INFO_TYPE_DEFINITIONS[label] ||
      INFO_TYPE_DEFINITIONS[coarse] || {
        title: `${label} Information Type`,
        text: "Definition not available yet."
      }
    );
  };

  return (
    <section className="rounded-xl border border-neutral-200 dark:border-neutral-800 p-4 space-y-4 min-w-0 min-h-0 h-full flex flex-col">
      {/* Show All Types Button */}
      <button
        onClick={() => setSelectedType(selectedType === "All" ? "All" : "All")}
        className={`w-full text-left space-y-1 rounded-lg p-1 ${
          selectedType === "All" ? "bg-blue-100 dark:bg-blue-900/30 border border-blue-300 dark:border-blue-700" : "hover:bg-neutral-50 dark:hover:bg-neutral-900"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 flex-1 min-w-0">
            <div className="text-sm font-semibold truncate">All Types</div>
            <div
              className="w-3 h-3 rounded-full bg-neutral-300 dark:bg-neutral-600 text-[8px] flex items-center justify-center text-white cursor-help"
              onMouseEnter={(e) => handleMouseEnter(e, ALL_TYPES_DEFINITION.title, ALL_TYPES_DEFINITION.text)}
              onMouseLeave={handleMouseLeave}
            >
              ?
            </div>
          </div>
          <div className={`flex-shrink-0 text-[10px] px-2 py-0.5 rounded-full border ${
            selectedType === "All"
              ? "border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30"
              : "border-neutral-300 text-neutral-500"
          }`}>
            {coverageByType.totalCoverage}%
          </div>
        </div>
        <div className="transition-all duration-500 ease-in-out">
          <ProgressBar 
            value={coverageByType.totalCoverage}
            variant="rainbow"
          />
          <div className="mt-1 text-[10px] text-neutral-500">Average across types</div>
        </div>
      </button>

      {/* Information Types Section */}
      <div className="space-y-3">
        {infoTypes.map((label) => {
          const v = coverageByType.byType[label] ?? 0;
          const color = getInfoTypeColor(label);
          const isSelected = selectedType === label;
          
          return (
            <div
              key={label}
              className={`w-full text-left space-y-1 rounded-lg p-1 border ${
                isSelected
                  ? "bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700"
                  : "border-transparent hover:bg-neutral-50 dark:hover:bg-neutral-900"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <button
                  onClick={() => setSelectedType(selectedType === label ? "All" : label)}
                  className="flex-1 min-w-0 text-left"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1 flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{label}</div>
                      <div
                        className="w-3 h-3 rounded-full bg-neutral-300 dark:bg-neutral-600 text-[8px] flex items-center justify-center text-white cursor-help"
                        onMouseEnter={(e) => {
                          handleMouseEnter(
                            e,
                            `${label} Coverage`,
                            PER_TYPE_COVERAGE_DEFINITION.text
                          );
                        }}
                        onMouseLeave={handleMouseLeave}
                      >
                        ?
                      </div>
                    </div>
                    <div className={`flex-shrink-0 text-[10px] px-2 py-0.5 rounded-full border ${
                      isSelected
                        ? "border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30"
                        : "border-neutral-300 text-neutral-500"
                    }`}>{v}%</div>
                  </div>
                  <div className="transition-all duration-500 ease-in-out mt-1">
                    <ProgressBar value={v} color={color} />
                  </div>
                </button>
                <button
                  className="mt-0.5 shrink-0 text-[10px] p-1 rounded-full border border-neutral-300 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                  onClick={() => {
                    setInfoTypes((prev) => {
                      const arr = Array.isArray(prev) ? prev : [];
                      const next = arr.filter((t) => t !== label);
                      return next;
                    });
                    if (selectedType === label) setSelectedType("All");
                  }}
                  aria-label={`Remove information type ${label}`}
                  title="Remove information type"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5">
                    <path d="M6 7a1 1 0 0 1 1-1h10a1 1 0 1 1 0 2H7a1 1 0 0 1-1-1z"/>
                  </svg>
                </button>
              </div>
            </div>
          );
        })}
        <button
          className="w-full text-xs rounded-lg border border-neutral-300 dark:border-neutral-700 px-3 py-2 hover:bg-neutral-50 dark:hover:bg-neutral-900"
          onClick={() => setIsAddOpen(true)}
        >
          + Add information type
        </button>
      </div>

      {/* Video Count and Facet Values Section */}
      <div className="space-y-2 flex-1">
        <div className="text-2xl font-bold transition-all duration-300 ease-in-out">
          {videoCount} videos
        </div>
        <div className="text-xs text-neutral-500">out of {totalVideos}</div>
        <div className="transition-all duration-300 ease-in-out flex items-center">
          <div className="text-[11px] flex flex-wrap items-center gap-2">
            {/* Show selected facets */}
            {Object.entries(selectedFacets).length > 0 && (
              <div className="text-neutral-500 animate-in fade-in-0 slide-in-from-top-1 duration-200">
                {Object.entries(selectedFacets).map(([k, v]) => (
                  <span key={k} className="mr-2">
                    {v.title}: <span className="font-medium text-neutral-700 dark:text-neutral-200">{v.label}</span>
                  </span>
                ))}
              </div>
            )}
            
            {/* Show placeholder when nothing is selected */}
            {Object.entries(selectedFacets).length === 0 && (
              <div aria-hidden className="h-4" />
            )}
          </div>
        </div>
      </div>
      {tooltip && <Tooltip tip={tooltip} />}

      {isAddOpen && (
        <div className="fixed inset-0 z-[1100] bg-black/30 flex items-center justify-center p-4" onMouseDown={() => setIsAddOpen(false)}>
          <div
            className="w-full max-w-lg rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 p-4 shadow-2xl"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold">Add an information type</div>
                <div className="text-xs text-neutral-500">Choose from available information types</div>
              </div>
              <button
                className="text-xs rounded-full border border-neutral-300 px-2 py-1 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                onClick={() => setIsAddOpen(false)}
              >
                Close
              </button>
            </div>

            <div className="mt-3">
              <input
                value={addQuery}
                onChange={(e) => setAddQuery(e.target.value)}
                placeholder="Search info types…"
                className="w-full text-sm rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2"
              />
            </div>

            {/* Placeholder for future stats in this add window */}
            <div className="mt-3 rounded-lg border border-dashed border-neutral-300 dark:border-neutral-700 p-3">
              <div className="text-xs font-medium text-neutral-700 dark:text-neutral-200">Stats (coming soon)</div>
              <div className="text-[11px] text-neutral-500 mt-1">We’ll show coverage / distribution info here in a later pass.</div>
            </div>

            <div className="mt-3 max-h-[50vh] overflow-y-auto space-y-2">
              {addableTypes.length === 0 ? (
                <div className="text-sm text-neutral-500 py-6 text-center">No information types to add.</div>
              ) : (
                addableTypes.map((t) => {
                  const def = getInfoTypeDefinition(t);
                  return (
                  <button
                    key={t}
                    className="w-full text-left rounded-lg border border-neutral-200 dark:border-neutral-800 p-3 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                    onClick={() => {
                      setInfoTypes((prev) => {
                        const arr = Array.isArray(prev) ? prev : [];
                        if (arr.includes(t)) return arr;
                        return [...arr, t];
                      });
                      setIsAddOpen(false);
                      setAddQuery("");
                    }}
                  >
                    <div className="text-sm font-medium">{t}</div>
                    <div className="text-xs text-neutral-500">{def.title}</div>
                    {String(def.text || "").trim() && (
                      <div className="mt-1 text-[11px] text-neutral-600 dark:text-neutral-400 line-clamp-3 whitespace-pre-wrap">
                        {def.text}
                      </div>
                    )}
                  </button>
                );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

