// UI constants for grid layout
// Note: Information types are now configured via datasetConfig.ts

// Grid template for 4-column layout: widen column 1, slightly reduce column 2
// Can be configured via NEXT_PUBLIC_GRID_TEMPLATE environment variable
export const GRID_TEMPLATE = process.env.NEXT_PUBLIC_GRID_TEMPLATE ?? "1.5fr 0.6fr 1fr 1fr";

// Number of information items to show when section is expanded
export const INFORMATION_WINDOW_EXPANDED_LIMIT: number = 3;

// Relevance threshold for showing information (0.0 to 1.0)
// Only pieces with at least this percentage of matching facets will be shown
export const INFORMATION_WINDOW_RELEVANCE_THRESHOLD: number = 0.7;

// Terminology dictionary for custom terms (information types, facet names, facet values)
// Maps original terms to display terms
export const UI_TERMINOLOGY: Record<string, string> = {
  // Information Type Display Names (content_type)
  'Explanation - Justification': 'Why do this?',
  'Supplementary - Tip': 'Helpful Tip',
  'Supplementary - Warning': 'Important Warning',
  'Method - Instruction': 'Step-by-Step Instruction',
  'Method - Subgoal': 'Subgoal',
  'Method - Tool': 'Tool Introduction',
  'Description - Status': 'Status',
  'Description - Context': 'Context',
  'Description - Tool Specification': 'Tool Specification',
  'Conclusion - Outcome': 'Final Result',
  'Conclusion - Reflection': 'Final Thoughts',
  
  // Facet Names (facet titles) - add mappings here as needed
  // Example: 'Actionability': 'Action Type',
  
  // Facet Values - add mappings here as needed
  // Example: 'Do': 'Perform Action',
  'na': 'N/A',
  'none': 'None',
};

// Helper function to get terminology for a term (information type, facet name, or facet value)
export function getTerminology(term: string): string {
  return UI_TERMINOLOGY[term] || term;
}

