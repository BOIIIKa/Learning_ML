import type { Lang } from "../../../i18n/ui";
import Plot2D from "../../viz/Plot2D";
import Widget from "../../ui/Widget";

const TEXT = {
  title: { zh: "例题：五个点的最大间隔", en: "Worked example: five points" },
  caption: {
    zh: "分离超平面 x₁ + x₂ = 1，虚线是间隔边界。三个带圈的点是支持向量，其中原点 x₄ 的乘子最大，尽管它对 w 的贡献为零。",
    en: "Separating hyperplane x₁ + x₂ = 1; dashed lines are the margin boundaries. The three ringed points are support vectors — the origin x₄ has the largest multiplier even though it contributes nothing to w.",
  },
};

/** The hand-solvable example of Section 4, drawn with its exact solution w = (1, 1), b = −1. */
export default function FivePoints({ lang }: { lang: Lang }) {
  const X = [[2, 0], [0, 2], [2, 2], [0, 0], [-1, 0]];
  const y = [1, 1, 1, -1, -1];
  return (
    <Widget title={TEXT.title[lang]} caption={TEXT.caption[lang]} tag="example">
      <Plot2D X={X} y={y} step={{ w: [1, 1], b: -1, support: [0, 1, 3] }} domain={{ x: [-1.8, 3], y: [-1.4, 3] }} height={400} tweenMs={0} />
    </Widget>
  );
}
