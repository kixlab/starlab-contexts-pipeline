// UI constants for grid layout
// Note: Information types are now configured via datasetConfig.ts

// Grid template for 4-column layout: widen column 1, slightly reduce column 2
// Can be configured via NEXT_PUBLIC_GRID_TEMPLATE environment variable
export const GRID_TEMPLATE = process.env.NEXT_PUBLIC_GRID_TEMPLATE ?? "1.5fr 0.6fr 1fr 1fr";

