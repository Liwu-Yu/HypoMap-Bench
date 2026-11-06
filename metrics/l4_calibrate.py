#!/usr/bin/env python3
"""L4 judge alignment with the experts (docs/specs/metrics.md §7).

    python metrics/l4_calibrate.py scores-luna.jsonl [scores-qwen.jsonl] [--output report.json]

Per judged dimension and for the overall mean of the four, over every
expert-rated workspace the judges scored, for each judge and for the panel
(mean of judges):
- alignment with the mean of the three experts: Spearman rho and Kendall
  tau-b (primary), Pearson r (supplementary);
- the human-human reference: each expert against the mean of the other two,
  averaged over the three experts, beside the judge against the same
  two-expert means (a like-for-like comparison);
- 95% bootstrap intervals (resampling the workspaces).
"average of four" averages each correlation over the four dimensions; it is the
number used to compare prompt versions and to decide on the panel.
Correlations use the judges' raw scores; a constant offset does not change them.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import warnings

import numpy as np

DIMS = ["hypothesis_clarity", "hierarchical_coherence", "traceability", "testability"]
METRICS = ["spearman", "kendall", "pearson"]
warnings.filterwarnings("ignore", "Mean of empty slice")  # a resample where every score ties


def ranks(v):
    """Ranks with ties sharing their mean rank."""
    v = np.round(np.asarray(v, float), 9)
    r = np.empty(len(v))
    r[v.argsort(kind="stable")] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        r[v == x] = r[v == x].mean()
    return r


def correlations(x, y):
    """Spearman, Kendall tau-b and Pearson; NaN when either side is constant."""
    x, y = np.round(np.asarray(x, float), 9), np.round(np.asarray(y, float), 9)
    if x.std() == 0 or y.std() == 0:
        return {m: float("nan") for m in METRICS}
    sx, sy = np.sign(x[:, None] - x[None, :]), np.sign(y[:, None] - y[None, :])
    return {"spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
            "kendall": float((sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())),
            "pearson": float(np.corrcoef(x, y)[0, 1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judges", type=Path, nargs="+", help="l4_judge.py outputs; files of one judge are pooled.")
    parser.add_argument("--experts", type=Path, required=True, help="Expert ratings JSONL (not released).")
    parser.add_argument("--participants", help="Comma-separated subset of participants (sensitivity checks).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    keep = set(args.participants.split(",")) if args.participants else None
    experts = [json.loads(line) for line in args.experts.open(encoding="utf-8")]
    experts = {r["artifact_id"]: r for r in experts if keep is None or r["participant"] in keep}
    expert_ids = sorted(next(iter(experts.values()))["scores"])

    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # judge -> workspace -> dimension -> scores
    for path in args.judges:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["parsed"]:
                for d, v in row["parsed"]["scores"].items():
                    if d in DIMS:
                        per[row["judge"]][row["workspace_id"]][d].append(v)
    judges = {name: {ws: {d: st.mean(v) for d, v in dims.items()} for ws, dims in rows.items() if len(dims) == len(DIMS)}
              for name, rows in per.items()}
    ids = sorted(set(experts).intersection(*judges.values()))
    if len(judges) > 1:
        judges["panel"] = {ws: {d: st.mean(judges[j][ws][d] for j in judges) for d in DIMS} for ws in ids}

    def table(score):  # dimension -> array over ids, with the overall mean of the four
        out = {d: np.array([score(ws, d) for ws in ids], float) for d in DIMS}
        out["overall"] = np.mean([out[d] for d in DIMS], axis=0)
        return out

    expert = {e: table(lambda ws, d, e=e: experts[ws]["scores"][e][d]) for e in expert_ids}
    target = {d: np.mean([expert[e][d] for e in expert_ids], axis=0) for d in DIMS + ["overall"]}
    others = {e: {d: np.mean([expert[o][d] for o in expert_ids if o != e], axis=0) for d in target}
              for e in expert_ids}
    scored = {name: table(lambda ws, d, name=name: judges[name][ws][d]) for name in judges}

    def statistics(sample):
        """All correlations for one set of workspace indices."""
        out = {}
        for d in target:
            out[d] = {name: correlations(x[d][sample], target[d][sample]) for name, x in scored.items()}
            out[d]["human-human"] = {m: np.nanmean([correlations(expert[e][d][sample], others[e][d][sample])[m]
                                                    for e in expert_ids]) for m in METRICS}
            for name, x in scored.items():
                out[d][f"{name} vs two experts"] = {
                    m: np.nanmean([correlations(x[d][sample], others[e][d][sample])[m] for e in expert_ids])
                    for m in METRICS}
        out["average of four"] = {row: {m: np.mean([out[d][row][m] for d in DIMS]) for m in METRICS}
                                  for row in out[DIMS[0]]}
        return out

    point = statistics(np.arange(len(ids)))
    rng = np.random.default_rng(0)
    boots = [statistics(rng.integers(0, len(ids), len(ids))) for _ in range(args.bootstrap)]
    report = {"n_workspaces": len(ids), "dimensions": {}}
    for d, rows in point.items():
        report["dimensions"][d] = {
            row: {m: {"value": float(v), "ci95": [float(np.nanpercentile([b[d][row][m] for b in boots], q))
                                                  for q in (2.5, 97.5)]}
                  for m, v in values.items()}
            for row, values in rows.items()}

    print(f"{len(ids)} workspaces; 95% bootstrap intervals over workspaces ({args.bootstrap} resamples).")
    for d, rows in report["dimensions"].items():
        print(f"\n{d}\n{'':28}" + "".join(f"{m.capitalize():>20}" for m in METRICS))
        for row, values in rows.items():
            label = row if "vs" in row or row == "human-human" else f"{row} vs expert mean"
            print(f"{label:28}" + "".join(
                f"{v['value']:>8.2f} [{v['ci95'][0]:>4.2f},{v['ci95'][1]:>5.2f}]" for v in values.values()))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
