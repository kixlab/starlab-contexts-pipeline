/** Position of a graph node. */
export type NodePosition = {
  x: number;
  y: number;
  layer: number;
  isDummy?: boolean;
};

/** Dataset loading status. */
export type DatasetStatus =
  | "idle"
  | "loading"
  | "computing"
  | "loaded"
  | "error";
