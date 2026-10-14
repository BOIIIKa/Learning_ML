// "Drive" scripts let a course step operate the widget on its slide, like a ValueTracker in manim.
//
//   s1=11          animate the 1st slider of the visual to 11 (2.5 s, smooth)
//   s2=0.3/4       … the 2nd slider to 0.3 over 4 s
//   s1=80%/3       … the 1st slider to 80 % of its range over 3 s
//   click:chip2    click the 2nd .chip (also btn3, check1, or any CSS selector)
//   wait:1.5       pause 1.5 s
// Commands run in order, separated by ";".
import type { Clock } from "./clock";
import { circumscribe } from "./animations";
import { smooth } from "./rate";

import { DRIVE_EVENT } from "./events";
export { DRIVE_EVENT };

export interface DriveCommand { kind: "slider" | "click" | "wait"; index?: number; to?: number; pct?: boolean; secs: number; target?: string }

export function parseDrive(spec: string): DriveCommand[] {
  return spec.split(";").map((s) => s.trim()).filter(Boolean).map((cmd): DriveCommand => {
    let m = cmd.match(/^s(\d+)\s*=\s*(-?[\d.]+)(%?)(?:\s*\/\s*([\d.]+))?$/);
    if (m) return { kind: "slider", index: +m[1], to: +m[2], pct: m[3] === "%", secs: m[4] ? +m[4] : 2.5 };
    m = cmd.match(/^wait:\s*([\d.]+)$/);
    if (m) return { kind: "wait", secs: +m[1] };
    m = cmd.match(/^click:\s*(.+)$/);
    if (m) return { kind: "click", target: m[1].trim(), secs: 0.9 };
    throw new Error(`Unknown drive command: ${cmd}`);
  });
}

/** Total scripted time in seconds (used to keep narration and animation in step). */
export const driveSeconds = (cmds: DriveCommand[]) => cmds.reduce((s, c) => s + c.secs, 0);

function resolveTarget(scope: Element, t: string): HTMLElement | null {
  const m = t.match(/^(chip|btn|check)(\d+)$/);
  if (!m) return scope.querySelector<HTMLElement>(t);
  const sel = { chip: ".chip, .chips button", btn: ".btn, button:not(.chip)", check: "input[type=checkbox]" }[m[1] as "chip" | "btn" | "check"];
  return [...scope.querySelectorAll<HTMLElement>(sel)][+m[2] - 1] ?? null;
}

export async function runDrive(clock: Clock, scope: HTMLElement, layer: HTMLElement, cmds: DriveCommand[], signal?: AbortSignal) {
  for (const c of cmds) {
    if (signal?.aborted) return;
    if (c.kind === "wait") await clock.wait(c.secs * 1000, signal);
    else if (c.kind === "slider") {
      const input = [...scope.querySelectorAll<HTMLInputElement>(".w-slider input[type=range]")][c.index! - 1];
      if (!input) { console.warn("drive: no slider", c.index); continue; }
      const from = +input.value;
      const to = c.pct ? +input.min + ((+input.max - +input.min) * c.to!) / 100 : c.to!;
      const row = input.closest(".w-slider") as HTMLElement | null;
      row?.classList.add("mj-driven");
      await clock.tween(c.secs * 1000, (a) => {
        input.dispatchEvent(new CustomEvent(DRIVE_EVENT, { detail: from + (to - from) * a }));
      }, smooth, signal);
      row?.classList.remove("mj-driven");
    } else {
      const el = resolveTarget(scope, c.target!);
      if (!el) { console.warn("drive: no target", c.target); continue; }
      const ring = circumscribe(clock, el, layer, { runTime: c.secs, signal });
      await clock.wait(c.secs * 400, signal);
      if (!signal?.aborted) el.click();
      await ring;
    }
  }
}
