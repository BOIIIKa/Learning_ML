// Main-thread client for the Pyodide worker. One shared worker per page; `stop()` kills it.
import type { Trace } from "../engines/trace";

export interface TestResult {
  name: { zh: string; en: string };
  passed: boolean;
  message: string;
}

export interface CheckResult {
  tests: TestResult[];
  traces: Trace[];
}

export type Status = "booting" | "packages" | "running";

const INDEX_URL: string = import.meta.env.PUBLIC_PYODIDE_URL || "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (r: CheckResult) => void; reject: (e: Error) => void; onStatus?: (s: Status) => void }>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./python.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const { id, status, done, result, error } = e.data;
    const job = pending.get(id);
    if (!job) return;
    if (status) job.onStatus?.(status);
    if (done) {
      pending.delete(id);
      if (error) job.reject(new Error(error));
      else job.resolve(result);
    }
  };
  return worker;
}

export function runCheck(
  job: { packages: string[]; files: Record<string, string>; student: string; check: string },
  onStatus?: (s: Status) => void,
): Promise<CheckResult> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onStatus });
    getWorker().postMessage({ id, indexURL: INDEX_URL, ...job });
  });
}

/** Start the interpreter and fetch packages in the background, so the first Run is quick.
 * Safe to call more than once: the worker boots once and loadPackage skips loaded packages. */
export function warm(packages: string[]) {
  getWorker().postMessage({ id: 0, warm: true, indexURL: INDEX_URL, packages });
}

/** Kill a runaway run (e.g. an infinite loop); the next run boots a fresh interpreter. */
export function stop() {
  worker?.terminate();
  worker = null;
  for (const [, job] of pending) job.reject(new Error("stopped"));
  pending.clear();
}
