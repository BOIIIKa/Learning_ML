// CAD-style crosshair: dashed guide lines to the axes plus a coordinate readout.
export function Crosshair({ p, W, H, label }: { p: [number, number] | null; W: number; H: number; label: string }) {
  if (!p) return null;
  const [x, y] = p;
  const right = x > W - 120;
  return (
    <g className="xhair" pointerEvents="none">
      <line x1={x} x2={x} y1={0} y2={H} />
      <line x1={0} x2={W} y1={y} y2={y} />
      <rect x={right ? x - 112 : x + 8} y={y < 26 ? y + 6 : y - 22} width={104} height={16} />
      <text x={right ? x - 106 : x + 14} y={y < 26 ? y + 18 : y - 10}>{label}</text>
    </g>
  );
}
