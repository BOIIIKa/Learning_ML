import { useMemo, useState } from "react";
import { gridFor, moons } from "../../../engines/data";
import { accuracy, forest, forestProba, growTree, predictProba } from "../../../engines/tree";
import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget, { Slider, Stat } from "../../ui/Widget";

const TEXT = {
  title: { zh: "一棵树太敏感，一片森林更稳", en: "One tree is jumpy; a forest is steady" },
  single: { zh: "单棵深树", en: "One deep tree" },
  forest: { zh: "随机森林", en: "Random forest" },
  trees: { zh: "树的数量", en: "Number of trees" },
  resample: { zh: "换一批训练数据", en: "Draw new training data" },
  test: { zh: "测试准确率", en: "Test accuracy" },
  caption: {
    zh: "反复点“换一批训练数据”：单棵深树的边界每次都剧烈跳动（高方差）；随机森林把许多在自助样本上训练、每次只看随机特征的树平均起来，边界几乎不动，测试准确率也更高。",
    en: "Keep pressing “draw new training data”: one deep tree's boundary jumps around every time (high variance). A random forest averages many trees, each trained on a bootstrap sample with random features, so its boundary barely moves and its test accuracy is higher.",
  },
};

export default function ForestPlayground({ lang }: { lang: Lang }) {
  const [seed, setSeed] = useState(0);
  const [mode, setMode] = useState<"single" | "forest">("single");
  const [n, setN] = useState(40);
  const train = useMemo(() => moons(100, 10 + seed, 0.32), [seed]);
  const test = useMemo(() => moons(400, 999, 0.32), []);
  const grid = useMemo(() => gridFor(test.X, 40, 0.3), [test]);

  const model = useMemo(() => {
    if (mode === "single") {
      const t = growTree(train.X, train.y, undefined, { maxDepth: 12 });
      return { proba: (x: number[]) => predictProba(t, x), acc: accuracy(t, test.X, test.y) };
    }
    const ts = forest(train.X, train.y, n, 7 + seed, 12);
    const proba = (x: number[]) => forestProba(ts, x);
    return { proba, acc: test.X.filter((x, i) => (proba(x) >= 0.5 ? 1 : -1) === test.y[i]).length / test.X.length };
  }, [mode, n, train, test]);

  const field = useMemo(() => grid.points.map((p) => model.proba(p) - 0.5), [model, grid]);

  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <div className="chips">
        {(["single", "forest"] as const).map((m) => (
          <button key={m} className={`chip${m === mode ? " on" : ""}`} onClick={() => setMode(m)}>{TEXT[m][lang]}</button>
        ))}
      </div>
      <Plot2D X={train.X} y={train.y} step={{ field }} grid={{ x: grid.x, y: grid.y }} tweenMs={400} />
      <div className="w-row">
        {mode === "forest" && <Slider label={TEXT.trees[lang]} value={n} min={1} max={100} step={1} onChange={setN} />}
        <button className="btn btn-primary" onClick={() => setSeed(seed + 1)}>↻ {TEXT.resample[lang]}</button>
      </div>
      <div className="w-stats">
        <Stat label={TEXT.test[lang]} value={`${(model.acc * 100).toFixed(1)}%`} tone="accent" />
      </div>
    </Widget>
  );
}
