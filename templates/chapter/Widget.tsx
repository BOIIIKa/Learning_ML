import { useMemo, useState } from "react";
import { fit, toyData } from "../../../engines/{{SLUG}}";
import type { Lang } from "../../../i18n/ui";
import CurvePlot from "../../viz/CurvePlot";
import Widget, { Slider, Stat } from "../../ui/Widget";

// One widget = one idea. Controls are Slider / chips / buttons so the animated course can drive
// them (drive="s1=…; click:chip2"). Colours come only from CSS variables (--accent, --accent-2,
// --class-pos, --class-neg, --ink, --text-muted) so both notes themes and the course palette work.
const TEXT = {
  title: { zh: "TODO(scaffold) 图题", en: "TODO(scaffold) Figure title" },
  n: { zh: "样本数", en: "Samples" },
  slope: { zh: "斜率", en: "Slope" },
  caption: {
    zh: "TODO(scaffold) 说明：读者看什么、拖什么、会发现什么。数据为合成示意。",
    en: "TODO(scaffold) What to look at, what to drag, what you will notice. Synthetic illustrative data.",
  },
};

export default function {{Widget}}({ lang }: { lang: Lang }) {
  const [n, setN] = useState(20);
  const data = useMemo(() => toyData(n, 1), [n]);
  const m = fit(data.xs, data.ys);
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="interactive">
      <CurvePlot xs={data.xs} ys={data.ys} xDomain={[0, 10]} yDomain={[-2, 24]}
        curves={[{ f: (x) => m.intercept + m.slope * x, color: "var(--class-pos)" }]} />
      <Slider label={TEXT.n[lang]} value={n} min={5} max={200} step={5} onChange={setN} />
      <div className="w-stats">
        <Stat label={TEXT.slope[lang]} value={m.slope.toFixed(3)} tone="accent" />
      </div>
    </Widget>
  );
}
