import type { TreeEdge } from "./utils/treeBuilder";

/** A visible edge — may have its source remapped to a visible ancestor. */
export type VisibleEdge = TreeEdge & { contracted?: boolean };

/** Position computed by stableLayout. */
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
