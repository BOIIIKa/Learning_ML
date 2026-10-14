export const LANGS = ["zh", "en"] as const;
export type Lang = (typeof LANGS)[number];
export type Bilingual = { zh: string; en: string };

export const pick = (text: Bilingual | string | undefined, lang: Lang): string =>
  text === undefined ? "" : typeof text === "string" ? text : text[lang];

export const otherLang = (lang: Lang): Lang => (lang === "zh" ? "en" : "zh");

/** Site-relative URL that respects Astro's `base`. */
export const href = (path: string) => `${import.meta.env.BASE_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;

const STRINGS = {
  siteName: { zh: "看得见的机器学习", en: "Visible ML" },
  tagline: { zh: "从严谨推导到科研前沿", en: "From rigorous derivations to the research frontier" },
  contents: { zh: "课程目录", en: "Contents" },
  handbook: { zh: "实战手册", en: "Handbook" },
  start: { zh: "开始学习", en: "Start learning" },
  available: { zh: "可学习", en: "Ready" },
  planned: { zh: "规划中", en: "Planned" },
  byPart: { zh: "按部分", en: "By part" },
  byTask: { zh: "按任务", en: "By task" },
  byStrategy: { zh: "按策略（损失）", en: "By strategy" },
  byAlgorithm: { zh: "按算法", en: "By algorithm" },
  chapter: { zh: "第 {n} 章", en: "Chapter {n}" },
  sources: { zh: "对应原书章节", en: "Source chapters" },
  prereqs: { zh: "前置", en: "Prerequisites" },
  goals: { zh: "本章目标", en: "Goals" },
  model: { zh: "模型", en: "Model" },
  strategy: { zh: "策略", en: "Strategy" },
  algorithm: { zh: "算法", en: "Algorithm" },
  frontier: { zh: "前沿跟进", en: "Frontier reading" },
  prev: { zh: "上一章", en: "Previous" },
  next: { zh: "下一章", en: "Next" },
  // Player
  play: { zh: "播放", en: "Play" },
  pause: { zh: "暂停", en: "Pause" },
  step: { zh: "第 {k} / {n} 步", en: "Step {k} / {n}" },
  speed: { zh: "速度", en: "Speed" },
  // Exercises
  exercises: { zh: "课后训练", en: "Practice" },
  derivationEx: { zh: "推导理解", en: "Derivations" },
  implementEx: { zh: "训练实战 · 补全算法", en: "Hands-on · Complete the algorithm" },
  libraryEx: { zh: "训练实战 · 组装库训练", en: "Hands-on · Assemble a library" },
  yourAnswer: { zh: "你的答案", en: "Your answer" },
  check: { zh: "检查", en: "Check" },
  correct: { zh: "正确！", en: "Correct!" },
  incorrect: { zh: "还不对，再想想", en: "Not quite — try again" },
  parseError: { zh: "无法解析表达式", en: "Could not parse the expression" },
  showSolution: { zh: "查看解答", en: "Show solution" },
  hideSolution: { zh: "收起解答", en: "Hide solution" },
  tabHint: { zh: "连按两次 Tab 填入参考答案", en: "Double-press Tab to fill in the reference answer" },
  tabAgain: { zh: "再按一次 Tab 填入答案", en: "Press Tab again to fill in the answer" },
  run: { zh: "运行并检查", en: "Run & check" },
  running: { zh: "运行中…", en: "Running…" },
  loadingPython: { zh: "正在浏览器中启动 Python（首次约需 10–30 秒）…", en: "Starting Python in your browser (first run takes ~10–30 s)…" },
  reset: { zh: "重置", en: "Reset" },
  fillAll: { zh: "填入全部答案", en: "Fill all answers" },
  allPassed: { zh: "全部通过！", en: "All tests passed!" },
  passedCount: { zh: "通过 {k} / {n}", en: "{k} / {n} passed" },
  traceTitle: { zh: "训练过程回放", en: "Training replay" },
  blanksLeft: { zh: "剩余空：{n}", en: "Blanks left: {n}" },
  runtimeError: { zh: "运行环境出错", en: "Runtime error" },
} satisfies Record<string, Bilingual>;

export type Key = keyof typeof STRINGS;

export function t(lang: Lang, key: Key, vars: Record<string, string | number> = {}): string {
  return STRINGS[key][lang].replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? `{${k}}`));
}

/** The six standard parts of every chapter, in order. */
export const SECTIONS = [
  { key: "scenario", zh: "情景引入", en: "Scenario" },
  { key: "method", zh: "方法", en: "The method" },
  { key: "math", zh: "数学工具与推导", en: "Math & derivations" },
  { key: "cases", zh: "案例与拓展", en: "Worked example & beyond" },
  { key: "practice", zh: "课后训练", en: "Practice" },
  { key: "handbook", zh: "实战手册与前沿", en: "Handbook & frontier" },
] as const;
export type SectionKey = (typeof SECTIONS)[number]["key"];
