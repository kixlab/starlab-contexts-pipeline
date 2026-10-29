// UI constants for information types and grid layout

function parseInfoTypesEnv(raw: string | undefined): string[] {
  if (!raw) return [];
  const trimmed = raw.trim();
  if (!trimmed) return [];

  // Allow JSON array: '["Method - Instruction","Supplementary - Tip"]'
  if (trimmed.startsWith("[")) {
    try {
      const v = JSON.parse(trimmed);
      if (Array.isArray(v)) return v.map(String).map((s) => s.trim()).filter(Boolean);
    } catch {
      // fall through
    }
  }

  // Default: CSV
  return trimmed
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

// Full `content_type` strings, preferably provided by `NEXT_PUBLIC_INFO_TYPES`.
export const INFO_TYPES: readonly string[] = parseInfoTypesEnv(process.env.NEXT_PUBLIC_INFO_TYPES);

// Convenience for UI dropdowns
export const INFO_TYPES_WITH_ALL: readonly string[] = ["All", ...INFO_TYPES];

// Grid template for 4-column layout: widen column 1, slightly reduce column 2
export const GRID_TEMPLATE = "1.5fr 0.6fr 1fr 1fr";

