// The training-trace format shared with python/learnml/trace.py.

export type Bilingual = { zh: string; en: string };

export interface TraceStep {
  w?: [number, number];
  b?: number;
  /** Decision values on the trace grid, row-major: field[r * nx + c] at (grid.x[c], grid.y[r]). */
  field?: number[];
  /** Per-sample weights (1 = average), drawn as point size. */
  weights?: number[];
  highlight?: number[];
  support?: number[];
  metrics?: Record<string, number>;
  note?: string;
}

export interface Trace {
  kind: "classifier2d";
  title?: Bilingual | string;
  X: number[][];
  y: number[];
  grid?: { x: number[]; y: number[] };
  steps: TraceStep[];
}
