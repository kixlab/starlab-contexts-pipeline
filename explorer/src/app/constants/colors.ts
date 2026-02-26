export const INFO_TYPE_COLORS = {
  Method: "#648FFF",
  Explanation: "#EEA400", 
  Supplementary: "#785EF0",
  Description: "#DC267F"
} as const;

export type InfoType = keyof typeof INFO_TYPE_COLORS;

// Helper function to get color for an info type
export function getInfoTypeColor(infoType: string): string {
  return INFO_TYPE_COLORS[infoType as InfoType] || "#6B7280"; // fallback to gray
}

// Helper function to determine if text should be white based on background color
export function shouldUseWhiteText(backgroundColor: string): boolean {
  // Convert hex to RGB
  const hex = backgroundColor.replace('#', '');
  const r = parseInt(hex.substr(0, 2), 16);
  const g = parseInt(hex.substr(2, 2), 16);
  const b = parseInt(hex.substr(4, 2), 16);
  
  // Calculate luminance
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  
  // Use white text if background is dark (luminance < 0.5)
  return luminance < 0.5;
}

// Color mapping for information categories in InfoSection
export const INFO_CATEGORY_COLORS: Record<string, { bg: string; border: string; header: string; glow: string }> = {
  'Method': {
    bg: 'rgba(255, 193, 7, 0.1)',
    border: 'rgba(255, 193, 7, 0.3)',
    header: '#FFC107',
    glow: 'rgba(255, 193, 7, 0.4)',
  },
  'Method - Instruction': {
    bg: 'rgba(255, 193, 7, 0.1)',
    border: 'rgba(255, 193, 7, 0.3)',
    header: '#FFC107',
    glow: 'rgba(255, 193, 7, 0.4)',
  },
  'Explanation': {
    bg: 'rgba(34, 197, 94, 0.1)',
    border: 'rgba(34, 197, 94, 0.3)',
    header: '#22C55E',
    glow: 'rgba(34, 197, 94, 0.4)',
  },
  'Explanation - Justification': {
    bg: 'rgba(34, 197, 94, 0.1)',
    border: 'rgba(34, 197, 94, 0.3)',
    header: '#22C55E',
    glow: 'rgba(34, 197, 94, 0.4)',
  },
  'Description': {
    bg: 'rgba(59, 130, 246, 0.1)',
    border: 'rgba(59, 130, 246, 0.3)',
    header: '#3B82F6',
    glow: 'rgba(59, 130, 246, 0.4)',
  },
  'Supplementary': {
    bg: 'rgba(239, 68, 68, 0.1)',
    border: 'rgba(239, 68, 68, 0.3)',
    header: '#EF4444',
    glow: 'rgba(239, 68, 68, 0.4)',
  },
  'Supplementary - Tip': {
    bg: 'rgba(239, 68, 68, 0.1)',
    border: 'rgba(239, 68, 68, 0.3)',
    header: '#EF4444',
    glow: 'rgba(239, 68, 68, 0.4)',
  },
  'Supplementary - Warning': {
    bg: 'rgba(239, 68, 68, 0.1)',
    border: 'rgba(239, 68, 68, 0.3)',
    header: '#EF4444',
    glow: 'rgba(239, 68, 68, 0.4)',
  },
};

// Default colors for categories not in the map (plain white fallback)
export const DEFAULT_CATEGORY_COLORS = {
  bg: 'rgba(255, 255, 255, 0.1)',
  border: 'rgba(255, 255, 255, 0.3)',
  header: '#FFFFFF',
  glow: 'rgba(255, 255, 255, 0.4)',
} as const;

// ---------------------------------------------------------------------------
// Dynamic color generation for content_types not in INFO_CATEGORY_COLORS.
// The hue is derived deterministically from the content_type string so that
// the same type always gets the same colour, regardless of order.
// ---------------------------------------------------------------------------

function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (Math.imul(31, hash) + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

export function generateCategoryColors(contentType: string): { bg: string; border: string; header: string; glow: string } {
  const hue = hashString(contentType) % 360;
  return {
    bg: `hsla(${hue}, 60%, 55%, 0.1)`,
    border: `hsla(${hue}, 60%, 55%, 0.3)`,
    header: `hsl(${hue}, 60%, 70%)`,
    glow: `hsla(${hue}, 60%, 55%, 0.4)`,
  };
}

/**
 * Return the colour palette for a given content_type.
 * Checks the hardcoded map first (atomic taxonomy labels), then falls back to
 * a deterministic generated palette so that any step label or "tutorial" also
 * gets a consistent, unique colour.
 */
export function getCategoryColors(contentType: string): { bg: string; border: string; header: string; glow: string } {
  return INFO_CATEGORY_COLORS[contentType] ?? generateCategoryColors(contentType);
}

// Semantic colors for UI elements
export const SEMANTIC_COLORS = {
  // Badge colors
  current: {
    bg: '#22C55E', // green-500
    text: '#FFFFFF',
  },
  otherVideo: {
    bg: '#F97316', // orange-500
    text: '#FFFFFF',
  },
  // Button colors (jump/play button)
  jumpButton: {
    bg: 'rgba(59, 130, 246, 0.3)', // blue-500/30
    border: 'rgba(59, 130, 246, 0.6)', // blue-500/60
    text: 'rgba(191, 219, 254, 1)', // blue-200
    hoverBg: 'rgba(59, 130, 246, 0.4)', // blue-500/40
    hoverBorder: 'rgba(59, 130, 246, 0.8)', // blue-500/80
  },
  // Facet colors
  differing: {
    bg: 'rgba(239, 68, 68, 0.2)', // red-500/20
    border: 'rgba(239, 68, 68, 0.5)', // red-500/50
    text: 'rgba(252, 165, 165, 1)', // red-300
    hoverBg: 'rgba(239, 68, 68, 0.3)', // red-500/30
  },
  matching: {
    bg: 'rgba(255, 255, 255, 0.05)',
    border: 'rgba(255, 255, 255, 0.1)',
    text: 'rgba(255, 255, 255, 0.5)',
    hoverBg: 'rgba(255, 255, 255, 0.1)',
  },
  // Tooltip colors
  tooltip: {
    differing: {
      piece: '#DC2626', // red-600
      current: '#2563EB', // blue-600
    },
    matching: '#16A34A', // green-600
  },
  // Background colors
  background: {
    current: 'rgba(255, 255, 255, 0.15)',
    default: 'rgba(255, 255, 255, 0.12)',
    border: 'rgba(255, 255, 255, 0.15)',
    shadow: 'rgba(0, 0, 0, 0.45)',
  },
} as const;