import type { DatasetVideo } from "@/app/types";

export interface DatasetOption {
  label: string;
  bundleUrl?: string;
  infoTypes: string[] | null;
  task?: string;
}

export const DATASET_OPTIONS: DatasetOption[] = [
  { label: "Synthetic paper folding", bundleUrl: "/data/demo-bundle.json", infoTypes: null },
  ...(process.env.NEXT_PUBLIC_BUNDLE_URL ? [{
    label: "Local pipeline output", bundleUrl: process.env.NEXT_PUBLIC_BUNDLE_URL, infoTypes: null,
  }] : []),
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
