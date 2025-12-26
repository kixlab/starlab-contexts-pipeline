// UI constants for grid layout
// Note: Information types are now configured via datasetConfig.ts

// Grid template for 4-column layout: widen column 1, slightly reduce column 2
// Can be configured via NEXT_PUBLIC_GRID_TEMPLATE environment variable
export const GRID_TEMPLATE = process.env.NEXT_PUBLIC_GRID_TEMPLATE ?? "1.5fr 0.6fr 1fr 1fr";

// Information types to display in InformationWindow (augment page)
// These determine which content_type sections are shown
export const INFORMATION_WINDOW_INFO_TYPES: string[] = [
    'Method - Instruction',
    'Explanation - Justification',
    'Supplementary - Tip',
    'Supplementary - Warning',
];

// Number of information items to show when section is expanded
export const INFORMATION_WINDOW_EXPANDED_LIMIT: number = 3;

// Relevance threshold for showing information (0.0 to 1.0)
// Only pieces with at least this percentage of matching facets will be shown
export const INFORMATION_WINDOW_RELEVANCE_THRESHOLD: number = 0.8;

