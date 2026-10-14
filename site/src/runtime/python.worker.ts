/// <reference lib="webworker" />
// Runs exercise checks in Pyodide, off the main thread.

interface Job {
  id: number;
  indexURL: string;
  packages: string[];
  files: Record<string, string>;
  student: string;
  check: string;
}

let pyodide: any = null;

async function boot(indexURL: string) {
  if (pyodide) return pyodide;
  const { loadPyodide } = await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL });
  return pyodide;
}

self.onmessage = async (event: MessageEvent<Job>) => {
  const job = event.data;
  const post = (msg: object) => (self as unknown as Worker).postMessage({ id: job.id, ...msg });
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
