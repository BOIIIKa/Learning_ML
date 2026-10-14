/// <reference lib="webworker" />
// Runs exercise checks in Pyodide, off the main thread.

interface Job {
  id: number;
  warm?: boolean;
  indexURL: string;
  packages: string[];
  files: Record<string, string>;
  student: string;
  check: string;
}

let booting: Promise<any> | null = null;

/** One interpreter per worker; a warm-up and a run arriving together share the same boot. */
function boot(indexURL: string): Promise<any> {
  booting ??= (async () => {
    const { loadPyodide } = await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`);
    return loadPyodide({ indexURL });
  })();
  booting.catch(() => (booting = null)); // let a later run retry after a failed download
  return booting;
}

self.onmessage = async (event: MessageEvent<Job>) => {
  const job = event.data;
  const post = (msg: object) => (self as unknown as Worker).postMessage({ id: job.id, ...msg });
  if (job.warm) {
    // Background warm-up: errors are ignored here and resurface on a real run.
    try { await (await boot(job.indexURL)).loadPackage(job.packages); } catch { /* ignore */ }
    return;
  }
  try {
    post({ status: "booting" });
    const py = await boot(job.indexURL);
    post({ status: "packages" });
    await py.loadPackage(job.packages);
    for (const [path, text] of Object.entries(job.files)) {
      const full = `/home/pyodide/lib/${path}`;
      py.FS.mkdirTree(full.slice(0, full.lastIndexOf("/")));
      py.FS.writeFile(full, text);
    }
    post({ status: "running" });
    py.globals.set("student_source", job.student);
    py.globals.set("check_source", job.check);
    const json = await py.runPythonAsync(`
import json, sys
if "/home/pyodide/lib" not in sys.path:
    sys.path.insert(0, "/home/pyodide/lib")
from learnml import checking
json.dumps(checking.run(student_source, check_source))
`);
    post({ done: true, result: JSON.parse(json) });
  } catch (err) {
    post({ done: true, error: String(err) });
  }
};
