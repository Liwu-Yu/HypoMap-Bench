"""Pooled test results of the seven models over all 479 cards and the 108 D1 cards, from rows.jsonl of this folder
(metrics/score_runs.py reports per family only). Same bootstrap as score_runs.py (2,000 paper resamples, seed 0);
paired differences against each baseline in BASELINES.

    python analysis/pooled.py
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data/results/final-seven"
sys.path.insert(0, str(ROOT / "metrics"))
from score_runs import bootstrap  # noqa: E402

BASELINES = ("sft-150", "Qwen/Qwen3-8B", "sft-v1")  # SFT+RL and others vs SFT; SFT vs base; SFT vs SFT w/o pairs
KEYS = ("valid", "complete", "parse", "schema_valid", "patch_applies", "l2", "l3", "coverage", "pitfall_rate",
        "l4", "l4_luna", "l4_qwen3-8b", "l4_hypothesis_clarity", "l4_hierarchical_coherence", "l4_traceability",
        "l4_testability", "d1_balanced_accuracy", "act_recall", "over_edit_rate", "pair_accuracy")

by_model = defaultdict(list)
for line in (HERE / "rows.jsonl").read_text().splitlines():
    r = json.loads(line)
    by_model[r["model"]].append(r)
out = {}
for scope, pick in (("all", lambda r: True), ("d1", lambda r: r["label"])):
    rows = {m: [r for r in rs if pick(r)] for m, rs in by_model.items()}
    reports = {b: bootstrap(rows, b) for b in BASELINES}
    out[scope] = {m: {k: {**reports[BASELINES[0]][m][k],
                          **{f"minus_{b}": reports[b][m][k][f"minus_{b}"] for b in BASELINES
                             if f"minus_{b}" in reports[b][m].get(k, {})}}
                      for k in KEYS if k in reports[BASELINES[0]][m]} for m in rows}
(HERE / "pooled.json").write_text(json.dumps(out, indent=1) + "\n")

fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
for scope, report in out.items():
    print(f"\n{scope:28}" + "".join(f"{m.split('/')[-1][:12]:>13}" for m in report))
    for k in KEYS:
        if k in report[BASELINES[0]]:
            print(f"  {k:26}" + "".join(f"{fmt(report[m][k]['value']):>13}" for m in report))
