// A pausable, speed-scaled clock shared by every animation of the course. Web Animations are
// registered with it so that pause / speed changes apply to them too; waits and JS tweens run
// on its virtual time.
import { smooth, type Rate } from "./rate";

type Waiter = { until: number; resolve: () => void };
type Ticker = (now: number) => boolean; // return false to unregister

export class Clock {
  speed = 1;
  paused = false;
  private virtual = 0;
  private last = performance.now();
  private waiters: Waiter[] = [];
  private tickers = new Set<Ticker>();
  private anims = new Set<Animation>();
  private raf = 0;

  constructor() {
    const loop = (t: number) => {
      const dt = t - this.last;
      this.last = t;
      if (!this.paused) this.virtual += dt * this.speed;
      const now = this.virtual;
      this.waiters = this.waiters.filter((w) => (w.until <= now ? (w.resolve(), false) : true));
      for (const tk of [...this.tickers]) if (!tk(now)) this.tickers.delete(tk);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  now() { return this.virtual; }

  /** Resolve after `ms` of clock time (pauses and speed apply). Aborting resolves immediately. */
  wait(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve) => {
      if (signal?.aborted || ms <= 0) return resolve();
      const w = { until: this.virtual + ms, resolve };
      this.waiters.push(w);
      signal?.addEventListener("abort", () => { this.waiters = this.waiters.filter((x) => x !== w); resolve(); }, { once: true });
    });
  }

  /** Drive `apply(alpha)` from 0 to 1 over `ms` with a rate function. Abort jumps to the end. */
  tween(ms: number, apply: (a: number) => void, rate: Rate = smooth, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve) => {
      const t0 = this.virtual;
      const finish = () => { apply(1); resolve(); };
      if (signal?.aborted || ms <= 0) return finish();
      let done = false;
      signal?.addEventListener("abort", () => { if (!done) { done = true; finish(); } }, { once: true });
      apply(rate(0));
      this.tickers.add((now) => {
        if (done) return false;
        const a = Math.min(1, (now - t0) / ms);
        apply(rate(a));
        if (a >= 1) { done = true; resolve(); return false; }
        return true;
      });
    });
  }

  /** Register a Web Animation so it follows pause and speed. Resolves when it finishes. */
  track(anim: Animation, signal?: AbortSignal): Promise<void> {
    anim.updatePlaybackRate(this.speed);
    if (this.paused) anim.pause();
    this.anims.add(anim);
    signal?.addEventListener("abort", () => { try { anim.finish(); } catch { /* infinite or idle */ } }, { once: true });
    return anim.finished.then(() => {}, () => {}).finally(() => this.anims.delete(anim));
  }

  setPaused(p: boolean) {
    this.paused = p;
    for (const a of this.anims) p ? a.pause() : a.play();
  }

  setSpeed(s: number) {
    this.speed = s;
    for (const a of this.anims) a.updatePlaybackRate(s);
  }

  /** Jump every running animation to its end state. */
  finishAll() {
    for (const a of this.anims) { try { a.finish(); } catch { /* ignore */ } }
  }

  dispose() { cancelAnimationFrame(this.raf); }
}
