// UI constants for information types and grid layout

export const INFO_TYPES = [
  "Method",
  "Explanation",
  "Description",
  "Supplementary",
] as const;

export const INFO_TYPES_WITH_ALL = ["All", ...INFO_TYPES] as const;

// Grid template for 4-column layout: widen column 1, slightly reduce column 2
export const GRID_TEMPLATE = "1.5fr 0.6fr 1fr 1fr";

