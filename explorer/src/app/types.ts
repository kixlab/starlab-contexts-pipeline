// The top-level kind of knowledge item in a dataset.
// - "atomic"  : a fine-grained information unit with a subtype from the information taxonomy
// - "step"    : a procedural step with a dataset-specific label as its subtype
// - "tutorial": a whole-tutorial summary with no subtype
export type KnowledgeItemType = "atomic" | "step" | "tutorial" | string;

export type DatasetVideoPiece = {
  piece_id: string;
  content: string;
  // item_type identifies the top-level knowledge category ("atomic" | "step" | "tutorial").
  // Pieces from old-format datasets that pre-date this field will have item_type undefined,
  // which should be treated as "atomic" by convention.
  item_type?: KnowledgeItemType;
  // content_type is the specific subtype / category label within item_type:
  //   atomic  → information-taxonomy label, e.g. "Method - Instruction"
  //   step    → dataset-specific step label, e.g. "gather ingredients, tools, and topping options"
  //   tutorial→ same as item_type ("tutorial"), since there is no meaningful subtype
  content_type: string;
  start: number;
  end: number;
  labels: Record<string, string[]>;
  unit_id: string;
};

export type TranscriptEntry = {
  text: string;
  start: number;
  end: number;
};

export type DatasetVideo = {
  id?: string;
  url: string;
  title: string;
  original_title: string;
  category?: string;
  content: string;
  transcript: TranscriptEntry[];
  pieces: DatasetVideoPiece[];
};

export type Facet = {
  type: string; // when | where | why
  title: string;
  title_plural: string;
  definition: string;
  guidelines?: string[];
  vocabulary: Array<{
    label: string;
    definition: string;
  }>;
  id: string;
};
