"""RL controls for the paper (2026-10-11): does GRPO in our pipeline move what its reward targets?

1. Test, from rows.jsonl of this folder (metrics/score_runs.py L0-L2, no judges): the earlier validity-gated run
   (grpo-50, from SFT step-125, reward docs/specs/metrics.md §10: invalid 0) against the paper's content-reward
   run (core-grpo-40, from SFT-150, core-content-v3), each paired with its own start. Same bootstrap as
   score_runs.py (2,000 paper resamples, seed 0).
2. Training logs of both runs (and of the 200-step continuation once it is back: add it to RUNS) (same pool, 16 cards x 4 rollouts, lr 1e-5, DAPO, no KL): the learning signal
   (groups whose four rewards are all equal give no gradient; spread within a group) and the reward of a card on
   its second draw against its first (paired by card; 2,000 card resamples).

    python analysis/controls.py
"""
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean, pstdev
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data/results/rl-controls"
OUT = ROOT / "analysis/out"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "metrics"))
from score_runs import bootstrap  # noqa: E402

PAIRS = (("grpo-50", "step-125"), ("core-grpo-40", "sft-150"), ("core-grpo-200", "sft-150"))
KEYS = ("valid", "strict_valid", "complete", "parse", "schema_valid", "patch_applies",
        "d1_balanced_accuracy", "act_recall", "over_edit_rate", "l3", "coverage", "pitfall_rate")
# the content runs, with Gemini correctness: data/results/core-grpo200 (the seven paper runs plus step 200)
CONTENT_ROWS = ROOT / "data/results/core-grpo200/rows.jsonl"
MAP_QUALITY = ROOT / "data/results/map-quality/scores-gemini.json"
DEV = ROOT / "data/results/rl-dev/core-grpo200"  # SFT-150 and steps 20-200, 217 cards x 2
RUNS = {"validity-gated (grpo-001)": (ROOT / "data/results/rl-training/grpo-001", "valid"),
        "content (core-grpo-40)": (ROOT / "data/results/rl-training/core-grpo40", "strict_L0"),
        "content, 200 steps (core-grpo200)": (ROOT / "data/results/rl-training/core-grpo200", "strict_L0")}

by_model = defaultdict(list)
for line in (HERE / "rows.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["model"] in ("step-125", "grpo-50"):
        by_model[r["model"]].append(r)
for line in CONTENT_ROWS.read_text().splitlines():
    r = json.loads(line)
    if r["model"] in ("sft-150", "core-grpo-40", "core-grpo-200"):
        by_model[r["model"]].append(r)
out = {"test": {}, "training": {}, "map_quality": {}, "development": {}}
mq = json.loads(MAP_QUALITY.read_text())["models"]
for m in ("core-grpo-40", "core-grpo-200"):
    out["map_quality"][m] = mq[m]["gain"]["minus_sft-150"]
for scope, pick in (("all", lambda r: True), ("d1", lambda r: r["label"])):
    for model, start in PAIRS:
        rep = bootstrap({m: [r for r in by_model[m] if pick(r)] for m in (start, model)}, start)
        out["test"][f"{scope}: {model} vs {start}"] = {
            k: {"start": rep[start][k]["value"], "rl": rep[model][k]["value"], "diff": rep[model][k][f"minus_{start}"]}
            for k in KEYS if k in rep[model] and rep[model][k]["value"] is not None}


def ci(xs, rng):
    means = sorted(mean(rng.choice(xs) for _ in xs) for _ in range(2000))
    return [round(means[50], 4), round(means[1949], 4)]


def dev_reward(m, judge):  # card -> mean core reward over the two development answers
    out = defaultdict(list)
    for s in (1, 2):
        for line in open(DEV / f"{m}-r{s}.core-{judge}.jsonl"):
            r = json.loads(line)
            out[r["task_id"]].append(r["reward"])
    return {t: mean(v) for t, v in out.items()}


for m in ("core-grpo-40", "core-grpo-200"):  # RL minus SFT-150, paired by card
    for judge in ("luna", "qwen"):
        a, b = dev_reward("sft-150", judge), dev_reward(m, judge)
        d = [b[t] - a[t] for t in sorted(a)]
        out["development"][f"{m} {judge}"] = {"value": round(mean(d), 4), "ci": ci(d, random.Random(0))}


for name, (folder, valid_key) in RUNS.items():
    rows = [json.loads(l) for rank in (0, 1) for l in open(folder / f"rewards-rank{rank}.jsonl")]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["step"], r["task_id"])].append(r)
    rewards = [[r["reward"] for r in g] for g in groups.values()]
    draws = defaultdict(dict)  # card -> step -> mean reward of its group(s) in that step
    for (step, task), g in groups.items():
        draws[task][step] = mean(r["reward"] for r in g)
    second = [s[sorted(s)[1]] - s[sorted(s)[0]] for s in draws.values() if len(s) >= 2]
    steps = max(r["step"] for r in rows) + 1
    windows = {f"{a}-{min(a + 10, steps) - 1}": [r for r in rows if a <= r["step"] < a + 10] for a in range(0, steps, 10)}
    out["training"][name] = {
        "steps": steps, "rollouts": len(rows), "groups": len(groups),
        "groups_all_equal": round(mean(float(max(v) == min(v)) for v in rewards), 4),
        "within_group_sd": round(mean(pstdev(v) for v in rewards), 4),
        "overall_sd": round(pstdev([r["reward"] for r in rows]), 4),
        "cards_drawn_twice": len(second),
        "second_minus_first_draw": {"value": round(mean(second), 4), "ci": ci(second, random.Random(0))},
        "kth_minus_first_draw": {k: {"cards": len(d), "value": round(mean(d), 4)} for k in range(2, 9) if (d := [
            s[sorted(s)[k - 1]] - s[sorted(s)[0]] for s in draws.values() if len(s) >= k])},  # the 200-step run
        "by_10_steps": {w: {"reward": round(mean(r["reward"] for r in rs), 3),
                            "valid": round(mean(float(bool(r[valid_key])) for r in rs), 3)} for w, rs in windows.items()},
    }
(OUT / "controls.json").write_text(json.dumps(out, indent=1) + "\n")

for name, rep in out["test"].items():
    print(f"\n{name}")
    for k, v in rep.items():
        d = v["diff"]
        print(f"  {k:22} {v['start']:.3f} -> {v['rl']:.3f}  {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
for name, rep in out["training"].items():
    print(f"\n{name}: " + json.dumps({k: v for k, v in rep.items() if k != "by_10_steps"}))
    print("  " + "  ".join(f"{w}: {v['reward']:.3f}/{v['valid']:.2f}" for w, v in rep["by_10_steps"].items()))

# Appendix table (analysis/out/tables/rl-controls.tex)
num = lambda v: ".00" if abs(v) < .005 else f"{v:+.2f}".replace("0.", ".")  # noqa: E731
diff = lambda d: f"${num(d['value'])}$ [${num(d['ci'][0])}$, ${num(d['ci'][1])}$]"  # noqa: E731
runs = [(out["test"][f"all: {m} vs {s}"], out["training"][t]) for (m, s), t in zip(PAIRS, RUNS)]
d1_runs = [out["test"][f"d1: {m} vs {s}"] for m, s in PAIRS]  # decision: resample the papers with decision cards, as in tables/ci.tex
content = ("core-grpo-40", "core-grpo-200")
lines = [
    "% RL controls; data/results/rl-controls/controls.json (generated by controls.py).",
    "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{6pt}",
    "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
    "& Validity-gated & Content (ours) & Content, continued \\\\", "\\midrule",
    "Start & SFT step 125 & SFT-150 & SFT-150 \\\\", "Steps & 50 & 40 & 200 \\\\", "\\addlinespace",
    "\\multicolumn{4}{@{}l}{\\emph{Test: RL $-$ start}} \\\\"]
for key, label in (("valid", "Validity"), ("patch_applies", "Patch applies"), ("strict_valid", "Strict validity"),
                   ("d1_balanced_accuracy", "Balanced accuracy"), ("l3", "Correctness"), ("mq", "Map quality")):
    tests = d1_runs if key == "d1_balanced_accuracy" else [t for t, _ in runs]
    cells = (["--"] + [diff(out["map_quality"][m]) for m in content] if key == "mq"
             else [diff(t[key]["diff"]) if key in t else "--" for t in tests])
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
lines += ["\\addlinespace", "\\multicolumn{4}{@{}l}{\\emph{Development: RL $-$ start}} \\\\",
          "Reward judge (GPT-6 Luna) & -- & " + " & ".join(diff(out["development"][f"{m} luna"]) for m in content) + " \\\\",
          "Independent judge (Qwen3-8B) & -- & " + " & ".join(diff(out["development"][f"{m} qwen"]) for m in content) + " \\\\",
          "\\addlinespace", "\\multicolumn{4}{@{}l}{\\emph{Training}} \\\\",
          "Tied groups & " + " & ".join(f"{100 * r['groups_all_equal']:.0f}\\%" for _, r in runs) + " \\\\",
          "Within-group SD & " + " & ".join(f"{r['within_group_sd']:.2f}".lstrip("0") for _, r in runs) + " \\\\",
          "2nd $-$ 1st draw & " + " & ".join(diff(r["second_minus_first_draw"]) for _, r in runs) + " \\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption{\\textbf{What RL moves.} Three GRPO runs on the same 789 tasks with the same settings: an earlier "
          "run whose reward scores an invalid answer 0, the run of Table~\\ref{tab:main}, and the same run continued to "
          "200 steps. Test: each run minus its starting adapter on all 479 tasks (balanced accuracy: the 108 decision "
          "cards). Development: the core reward on 217 tasks with two answers each, from the reward judge "
          "and from base Qwen3-8B, which takes no part in training. 95\\% intervals; --: not scored. Tied groups: share of training "
          "groups whose four rewards are equal (no gradient); within-group SD: mean standard deviation of the four "
          "rewards; 2nd $-$ 1st draw: reward of a task on its second draw in training minus its first.}",
          "\\label{tab:rl-controls}", "\\end{table*}"]
(OUT / "tables/rl-controls.tex").write_text("\n".join(lines) + "\n")
