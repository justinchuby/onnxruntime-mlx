"""Diff two benchmark result JSONs (base vs PR) into a Markdown perf report.

Emits a PR-comment-ready table flagging per-case regressions/improvements beyond a
threshold, plus any claim-status changes (an op that newly falls back to CPU is the
loudest kind of perf regression). Prints the Markdown to stdout; CI upserts it as a
sticky PR comment keyed on the hidden marker below.

Usage:
    python bench/compare.py --base base.json --pr pr.json [--threshold 10] > comment.md
"""

from __future__ import annotations

import argparse
import json

MARKER = "<!-- mlx-ep-bench -->"


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _index(data: dict) -> dict:
    return {c["name"]: c for c in data.get("cases", [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="flag as regression/improvement when |Δ%%| exceeds this")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit non-zero if any timing regression or new CPU fallback is found")
    args = ap.parse_args()

    base, pr = _load(args.base), _load(args.pr)
    bi, pi = _index(base), _index(pr)
    names = sorted(set(bi) | set(pi))

    rows = []
    regressions = improvements = fallbacks = 0
    for name in names:
        b, p = bi.get(name), pi.get(name)
        if b is None:
            rows.append((name, "—", "new", 0.0, "🆕 new case", 1e9))
            continue
        if p is None:
            rows.append((name, "removed", "—", 0.0, "🗑 removed", -1e9))
            continue
        bm, pm = b["mlx_ms"], p["mlx_ms"]
        delta = (pm - bm) / bm * 100.0 if bm > 0 else 0.0
        note = ""
        # Claim-status change dominates: a newly-declined op is a hard regression.
        if b.get("claimed") and not p.get("claimed"):
            note = f"⛔ now falls back to CPU ({', '.join(p.get('unclaimed') or [])})"
            fallbacks += 1
        elif not b.get("claimed") and p.get("claimed"):
            note = "🎉 now fully claimed (was CPU fallback)"
        elif delta > args.threshold:
            note = "🔴 regression"
            regressions += 1
        elif delta < -args.threshold:
            note = "🟢 improvement"
            improvements += 1
        rows.append((name, f"{bm:.3f}", f"{pm:.3f}", delta, note, delta))

    rows.sort(key=lambda r: r[5], reverse=True)  # worst regressions first

    out = [MARKER, "## 🏎️ MLX EP op benchmark", ""]
    parts = []
    if fallbacks:
        parts.append(f"**{fallbacks} new CPU fallback(s)** ⛔")
    if regressions:
        parts.append(f"{regressions} regression(s) 🔴")
    if improvements:
        parts.append(f"{improvements} improvement(s) 🟢")
    summary = " · ".join(parts) if parts else "no significant change"
    out.append(f"Median `session.run` (MLX EP), base vs PR — {summary} "
               f"(threshold ±{args.threshold:.0f}%).")
    out.append("")
    out.append("| Case | base ms | PR ms | Δ% | note |")
    out.append("|------|--------:|------:|---:|------|")
    for name, bms, pms, delta, note, _ in rows:
        d = f"{delta:+.1f}%" if bms not in ("—", "removed") and pms not in ("—",) else "—"
        out.append(f"| `{name}` | {bms} | {pms} | {d} | {note} |")
    out.append("")
    out.append(f"<sub>base `{base.get('label')}` · PR `{pr.get('label')}` · "
               f"{pr.get('iters')} iters (median) · ORT {pr.get('ort_version')} · "
               f"lower ms is better. Timings on a shared runner are noisy; only |Δ%| > "
               f"{args.threshold:.0f}% is flagged.</sub>")
    print("\n".join(out))

    if args.fail_on_regression and (regressions or fallbacks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
