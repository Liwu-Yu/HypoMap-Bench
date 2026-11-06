#!/usr/bin/env python3
"""Score the named-concern decision (D1) of generated development answers, per checkpoint.

D1 is act when the answer makes the concern operation on the named target:
- evidence_audit / local_repair: REFINE_ASPECT or RELINK_EVIDENCE whose target_ids
  include the first ID named in the question;
- required_aspect_completion: any ADD_ASPECT;
- guided_preservation: any proposed update.
On the development and test references this rule reproduces every saved D1 label.
A missing or unparsable answer counts as wrong for both outcomes.

    python metrics/score_d1.py selection/results/*.jsonl
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"\b([a-z]{1,3}\d{1,3}[a-z]?)\b")
CLAUSE = "Change the named target only"
FAMILIES = ("evidence_audit", "local_repair", "required_aspect_completion", "guided_preservation")


def predict(family, question, answer):
    if not isinstance(answer, dict) or not isinstance(answer.get("proposed_updates"), list):
        return "invalid"
    updates = [u for u in answer["proposed_updates"] if isinstance(u, dict)]
    if family in ("evidence_audit", "local_repair"):
        ids = ID.findall(question.split(CLAUSE)[0])
        target = ids[0] if ids else None
        hit = any(u.get("operation") in ("REFINE_ASPECT", "RELINK_EVIDENCE")
                  and (target is None or target in (u.get("target_ids") or [])) for u in updates)
    elif family == "required_aspect_completion":
        hit = any(u.get("operation") == "ADD_ASPECT" for u in updates)
    else:
        hit = bool(updates)
    return "act" if hit else "no-act"


def balanced(rows):
    recalls = []
    for label in ("act", "no-act"):
        group = [r for r in rows if r["d1"] == label]
        if group:
            recalls.append(sum(r["pred"] == label for r in group) / len(group))
    return sum(recalls) / len(recalls) if recalls else None


def score(rows):
    act = [r for r in rows if r["d1"] == "act"]
    no_act = [r for r in rows if r["d1"] == "no-act"]
    by_id = {r["card_id"]: r for r in rows}
    pairs = [(r, by_id[r["parent_task_id"]]) for r in rows
             if r["is_variant"] and r["parent_task_id"] in by_id]
    return {
        "invalid": sum(r["pred"] == "invalid" for r in rows),
        "balanced_accuracy": balanced(rows),
        "act_recall": sum(r["pred"] == "act" for r in act) / len(act),
        "over_edit_rate": sum(r["pred"] == "act" for r in no_act) / len(no_act),
        "pair_accuracy": sum(a["pred"] == a["d1"] and b["pred"] == b["d1"] for a, b in pairs) / len(pairs),
        "families": {f: balanced([r for r in rows if r["task_family"] == f]) for f in FAMILIES},
    }


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+",
                        help="Inference JSONL files; several files of one model are averaged as samples.")
    parser.add_argument("--references", type=Path, default=ROOT / "selection/dev-d1-references.jsonl")
    parser.add_argument("--run_metadata", type=Path,
                        default=ROOT / "SFT_training/outputs/run-002/run_metadata.json")
    parser.add_argument("--output", type=Path, default=ROOT / "selection/d1-scores.json")
    parser.add_argument("--no_select", action="store_true", help="Report only; no checkpoint selection (test set).")
    args = parser.parse_args()

    refs = {}
    with args.references.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            refs[row["card_id"]] = row
    losses = {}
    if args.run_metadata.exists():
        losses = {int(k): v for k, v in json.loads(args.run_metadata.read_text())
                  .get("eval_loss_by_step", {}).items()}

    samples = defaultdict(list)
    for path in args.results:
        answers, model = {}, None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                model = row["requested_model"]
                answers[row["card_id"]] = row.get("parsed_answer")
        missing = set(refs) - set(answers)
        if missing:
            raise ValueError(f"{path}: {len(missing)} development D1 cards have no result")
        rows = [{**ref, "pred": predict(ref["task_family"], ref["question"], answers[cid])}
                for cid, ref in refs.items()]
        samples[model].append({"result_file": str(path), **score(rows)})

    report = {}
    for model, runs in samples.items():
        step = int(model.split("-")[-1]) if model.startswith("step-") else (0 if model == "Qwen/Qwen3-8B" else None)
        bal = [r["balanced_accuracy"] for r in runs]
        report[model] = {
            "step": step, "eval_loss": losses.get(step) if step is not None else None, "samples": len(runs),
            **{k: mean([r[k] for r in runs]) for k in
               ("balanced_accuracy", "act_recall", "over_edit_rate", "pair_accuracy")},
            "balanced_accuracy_range": [min(bal), max(bal)],
            "invalid": sum(r["invalid"] for r in runs),
            "families": {f: mean([r["families"][f] for r in runs]) for f in FAMILIES},
            "runs": runs,
        }

    ranked = sorted(report.items(), key=lambda kv: (kv[1]["step"] is None, kv[1]["step"] or 0, kv[0]))
    print(f"{'model':<16}{'n':>3}{'bal.acc':>8}{'range':>12}{'act-rec':>8}{'over-ed':>8}{'pair':>7}"
          f"{'invalid':>8}{'dev loss':>9}  " + "  ".join(f[:6] for f in FAMILIES))
    for name, s in ranked:
        lo, hi = s["balanced_accuracy_range"]
        fam = "  ".join(f"{s['families'][f]:.2f}  " for f in FAMILIES)
        loss = f"{s['eval_loss']:.4f}" if s["eval_loss"] is not None else "-"
        print(f"{name:<16}{s['samples']:>3}{s['balanced_accuracy']:>8.3f}{f'{lo:.2f}-{hi:.2f}':>12}"
              f"{s['act_recall']:>8.3f}{s['over_edit_rate']:>8.3f}{s['pair_accuracy']:>7.3f}"
              f"{s['invalid']:>8}{loss:>9}  {fam}")

    # Selection: highest mean balanced accuracy among trained checkpoints. Within
    # 0.03 of the best (about one or two cards), prefer the lower development loss.
    trained = [] if args.no_select else [(n, s) for n, s in ranked if n.startswith("step-")]
    if trained:
        best = max(s["balanced_accuracy"] for _, s in trained)
        near = [(n, s) for n, s in trained if s["balanced_accuracy"] >= best - 0.03]
        choice = min(near, key=lambda ns: (ns[1]["eval_loss"] if ns[1]["eval_loss"] is not None else 9,
                                           -ns[1]["balanced_accuracy"]))[0]
        print(f"\nSelected checkpoint: {choice} (best mean balanced accuracy {best:.3f}; "
              f"candidates within 0.03: {', '.join(n for n, _ in near)})")
        report["_selected"] = choice
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
