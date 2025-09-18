export type DatasetVideoPiece = {
  piece_id: string;
  content: string;
  content_type: "Method" | "Explanation" | "Description" | "Supplementary" | string;
  start: number;
  end: number;
  labels: Record<string, string[]>;
  unit_id: string;
};

export type DatasetVideo = {
  url: string;
  title: string;
  original_title: string;
  pieces: DatasetVideoPiece[];
};

export type Facet = {
  type: string; // when | where | why
  title: string;
  title_plural: string;
  question: string;
  answer_guidelines?: string[];
  answers: Array<{
    answer: string;
    definition: string;
  }>;
};
