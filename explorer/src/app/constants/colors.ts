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