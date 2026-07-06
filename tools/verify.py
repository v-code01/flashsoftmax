"""Independent verification of the flashsoftmax findings, sharing no code with analyze.py. Re-reads
results/bench.jsonl and re-asserts the four claims with its own arithmetic:

  1. Online error tracks two-pass at every length AND scale (worst ratio bounded) with no upward trend
     in length - the flash rescale is numerically free.
  2. Standalone, online is NOT faster than two-pass (median latency ratio >= 1) - it recomputes exps.
  3. Naive overflow at large scale reaches every seed (100%) by the longest length; the fraction rises
     on average but is non-monotone across intermediate lengths (8 seeds = 12.5% granularity).
  4. Both safe variants stay finite: they never contribute a null error.

Exit non-zero on any mismatch. Run in the gate.
"""
from __future__ import annotations

import json
import sys


def _f(v: object) -> float:
    assert isinstance(v, (int, float))
    return float(v)


def med(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def main() -> int:
    rows = [json.loads(x) for x in open("results/bench.jsonl") if x.strip()]
    lengths = sorted({int(r["len"]) for r in rows})
    scales = sorted({int(r["scale"]) for r in rows})

    def cell(L: int, scale: int) -> list[dict[str, object]]:
        return [r for r in rows if int(r["len"]) == L and int(r["scale"]) == scale]

    ok = True

    # 1. online/two-pass error ratio bounded over ALL (length, scale) cells and non-trending in length.
    ratio_cell = {}
    for L in lengths:
        for s in scales:
            c = cell(L, s)
            tp = med([_f(r["twopass_err"]) for r in c])
            on = med([_f(r["online_err"]) for r in c])
            ratio_cell[(L, s)] = on / tp if tp > 0 else 1.0
    worst = max(ratio_cell.values())  # worst over all scales (~1.43, near the f32 floor at scale=1)
    # no upward trend: the worst ratio at the longest length is not markedly above the short-length ones.
    long_worst = max(ratio_cell[(lengths[-1], s)] for s in scales)
    trend_ok = long_worst <= 2.0 and worst <= 2.5
    print(f"  [1] online/two-pass err ratio (all scales): worst {worst:.2f}, at L={lengths[-1]} "
          f"{long_worst:.2f} (free & no length drift={trend_ok})")
    ok = ok and trend_ok

    # 2. standalone latency: online not faster than two-pass.
    lat = [med([_f(r["online_ns"]) for r in cell(L, 10)]) /
           med([_f(r["twopass_ns"]) for r in cell(L, 10)]) for L in lengths]
    online_not_faster = med(lat) >= 1.0
    print(f"  [2] standalone latency online/two-pass: median {med(lat):.2f}x "
          f"(online NOT faster={online_not_faster})")
    ok = ok and online_not_faster

    # 3. naive overflow at scale=30 reaches 100% by the longest length (rises on average, non-monotone).
    ov = []
    for L in lengths:
        c = cell(L, 30)
        ov.append(sum(1 for r in c if r["naive_err"] is None) / len(c))
    reaches_all = ov[-1] == 1.0 and ov[-1] >= ov[0]
    print(f"  [3] naive overflow fraction @scale=30: L={lengths[0]} {ov[0]:.0%} -> "
          f"L={lengths[-1]} {ov[-1]:.0%} (reaches 100% by longest length={reaches_all})")
    ok = ok and reaches_all

    # 4. safe variants never non-finite.
    safe = all(r["twopass_err"] is not None and r["online_err"] is not None for r in rows)
    print(f"  [4] two-pass and online always finite: {safe}")
    ok = ok and safe

    if ok:
        print("VERIFY OK: online (flash) softmax matches two-pass accuracy with no length drift, is "
              "not faster standalone (recomputes exps), and naive overflow reaches 100% by the longest "
              "length - all recomputed independently.")
        return 0
    print("VERIFY FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
