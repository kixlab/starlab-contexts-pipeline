export type DatasetVideoPiece = {
  piece_id: string;
  content: string;
  content_type: "Method" | "Explanation" | "Description" | "Supplementary" | string;
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
  url: string;
  title: string;
  original_title: string;
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
};
