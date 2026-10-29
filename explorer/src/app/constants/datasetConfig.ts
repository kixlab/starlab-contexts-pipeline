import type { DatasetVideo } from "@/app/types";

export interface DatasetOption {
  label: string;
  bundleUrl?: string;
  infoTypes: string[] | null;
  task?: string;
}

function configuredDatasets(): DatasetOption[] {
  const options: DatasetOption[] = [];
  if (process.env.NEXT_PUBLIC_BUNDLE_URL) options.push({
    label: process.env.NEXT_PUBLIC_DATASET_LABEL || "Local dataset",
    bundleUrl: process.env.NEXT_PUBLIC_BUNDLE_URL, infoTypes: null,
  });
  if (process.env.NEXT_PUBLIC_DATASETS) {
    const entries: unknown = JSON.parse(process.env.NEXT_PUBLIC_DATASETS);
    if (!Array.isArray(entries)) throw new Error("NEXT_PUBLIC_DATASETS must be a JSON array");
    for (const entry of entries) {
      if (!entry || typeof entry.label !== "string" || !entry.label.trim() ||
          typeof entry.bundleUrl !== "string" || !entry.bundleUrl.trim()) {
        throw new Error("Each NEXT_PUBLIC_DATASETS entry needs label and bundleUrl strings");
      }
      options.push({label: entry.label, bundleUrl: entry.bundleUrl, infoTypes: null});
    }
  }
  const labels = options.map(option => option.label);
  if (new Set([...labels, "Synthetic paper folding"]).size !== labels.length + 1) {
    throw new Error("Dataset labels must be unique");
  }
  return options;
}

export const DATASET_OPTIONS: DatasetOption[] = [
  ...configuredDatasets(),
  { label: "Synthetic paper folding", bundleUrl: "/data/demo-bundle.json", infoTypes: null },
];
export const DEFAULT_DATASET_OPTION = DATASET_OPTIONS[0];

export function deriveInfoTypesFromDataset(dataset: DatasetVideo[]): string[] {
  return Array.from(new Set(dataset.flatMap(v => v.pieces.map(p => p.content_type)).filter(Boolean))).sort();
}

export function getSelectedDataset(datasetLabel?: string): DatasetOption {
  if (typeof window === "undefined") return DEFAULT_DATASET_OPTION;
  const label = datasetLabel ?? new URLSearchParams(window.location.search).get("dataset");
  return DATASET_OPTIONS.find(option => option.label === label) ?? DEFAULT_DATASET_OPTION;
}
