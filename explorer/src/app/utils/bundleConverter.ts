/**
 * bundleConverter.ts
 *
 * Detects the "step" (new) bundle format by the presence of an `iterations`
 * key and converts it on-the-fly to the legacy "old" format expected by the
 * rest of the application.
 *
 * Step format  →  Old format
 * ─────────────────────────────────────────────────────────────────────────
 * facets            → context_schema + facet_candidates (same list)
 * knowledge_base    → labeled_dataset[].pieces  (one piece per step item)
 * annotations       → piece.labels
 * iterations        → (dropped)
 * frontier          → (dropped)
 */

import type { DatasetVideo, DatasetVideoPiece, Facet } from "@/app/types";

// ---------------------------------------------------------------------------
// Step-format internal types
// ---------------------------------------------------------------------------

type StepVocabItem = {
  id?: string;
  label: string;
  definition: string;
};

type StepFacet = {
  id: string;
  title: string;
  title_plural: string;
  scope: string;
  definition: string;
  justification?: string;
  guidelines?: string[];
  vocabulary: StepVocabItem[];
  metadata?: Record<string, unknown>;
};

type KBItemSource = {
  video_id: string;
  local_index: number;
  time_range: [number, number];
  parent_id: string | null;
  source_type: string;
};

type KBItem = {
  id: string;
  source: KBItemSource;
  text: string;
  type: string;
  subtype: string;
  unit_id: string;
  metadata?: Record<string, unknown>;
};

type StepBundle = {
  facets: StepFacet[];
  frontier: unknown[];
  annotations: [string, string, string][];
  knowledge_base: { items: KBItem[] };
  iterations: number;
};

// ---------------------------------------------------------------------------
// Old-format output shape (what the app expects)
// ---------------------------------------------------------------------------

type OldBundle = {
  context_schema: Facet[];
  facet_candidates: Facet[];
  labeled_dataset: DatasetVideo[];
};

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

export function isStepFormat(bundle: Record<string, unknown>): bundle is StepBundle {
  return "iterations" in bundle;
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

function convertFacet(sf: StepFacet): Facet {
  return {
    id: sf.id,
    // 'scope' (global / mid-level / local) maps to the 'type' field in Facet
    type: sf.scope ?? "local",
    title: sf.title,
    title_plural: sf.title_plural,
    definition: sf.definition,
    guidelines: sf.guidelines ?? [],
    // Strip the optional 'id' from each vocabulary entry
    vocabulary: sf.vocabulary.map(({ label, definition }) => ({ label, definition })),
  };
}

// ---------------------------------------------------------------------------
// Main normalizer
// ---------------------------------------------------------------------------

/**
 * Accepts any raw bundle JSON object and returns it in the old format.
 * If the bundle is already in the old format it is returned unchanged.
 */
export function normalizeBundle(bundle: Record<string, unknown>): OldBundle {
  // Already old format — pass through
  if (!isStepFormat(bundle)) {
    return bundle as unknown as OldBundle;
  }

  const step = bundle as StepBundle;

  // 1. Build annotation index: itemId → { facetId: [label] }
  const annIndex = new Map<string, Record<string, string[]>>();
  for (const [itemId, facetId, label] of step.annotations) {
    if (!annIndex.has(itemId)) annIndex.set(itemId, {});
    annIndex.get(itemId)![facetId] = [label];
  }

  // 2. Convert facets (used for both context_schema and facet_candidates)
  const convertedFacets: Facet[] = step.facets.map(convertFacet);

  // 3. Group KB items by video_id
  const byVideo = new Map<string, KBItem[]>();
  for (const item of step.knowledge_base.items) {
    const vid = item.source.video_id;
    if (!byVideo.has(vid)) byVideo.set(vid, []);
    byVideo.get(vid)!.push(item);
  }

  // 4. Build labeled_dataset
  const labeled_dataset: DatasetVideo[] = [];
  let videoIdx = 0;

  for (const [videoId, items] of byVideo) {
    // Keep steps in source order
    items.sort((a, b) => a.source.local_index - b.source.local_index);

    const pieces: DatasetVideoPiece[] = items.map((item, pieceIdx) => ({
      piece_id: `piece_${videoIdx}_${pieceIdx}`,
      unit_id: item.unit_id,
      start: item.source.time_range[0],
      end: item.source.time_range[1],
      // 'subtype' is the short step label; 'text' is the longer narrative
      content: item.subtype,
      content_type: item.subtype,
      labels: annIndex.get(item.id) ?? {},
    }));

    labeled_dataset.push({
      id: videoId,
      url: `https://www.youtube.com/watch?v=${videoId}`,
      title: videoId,
      original_title: videoId,
      content: items.map((i) => i.text).join(" "),
      transcript: [],
      pieces,
    });

    videoIdx++;
  }

  return {
    context_schema: convertedFacets,
    facet_candidates: convertedFacets,
    labeled_dataset,
  };
}
