#!/usr/bin/env python3
"""GRPO reward (docs/specs/metrics.md §10), for training and for scoring generated answers.

    invalid answer (L0)   0
    D1 cards              (2·D1 + L2 + L3) / 4
    other cards           (L2 + L3) / 2

- L0: the generation finished, the final answer follows the thinking and parses as JSON, and
  `validate_answer` passes (schema, and the patch applies under the public contract).
- D1: the named-concern rule of metrics/score_d1.py; 1 if it matches the card's label.
- L2: 1 unless the patch changes an existing object outside the question's scope. On a card whose
  question or guidance names IDs, the scope is those objects and everything below them through
  composition links; a replaced node, or a replaced or removed relationship touching nothing in
  scope, is a violation. (`validate_answer` already confines every change to the update's own
  targets and the allowed operations.)
- L3: metrics/l3_judge.py, one Luna call; only valid answers are judged. A failed call takes the
  mean L3 of the card's other rollouts in the batch, so it adds no advantage.

Score generated answers (inference.py output) against a cards file written by package.py:

    python metrics/reward.py --cards selection/cards.jsonl selection/results/*.jsonl
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "metrics")]

from hypomap.grading import validate_answer  # noqa: E402
from hypomap.tasks import object_ids  # noqa: E402
from l3_judge import judge  # noqa: E402
from score_d1 import balanced, predict  # noqa: E402

JUDGE = "gpt-6-luna"


def extract(text):
    """The final answer after the thinking, as in inference.py; None if absent or not JSON."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def named_ids(packet):
    prose = f"{packet.get('question') or ''} {packet.get('user_guidance') or ''}"
    return {i for i in object_ids(packet) if re.search(rf"(?<![\w-]){re.escape(i)}(?![\w-])", prose)}


def l2(packet, answer):
    scope = named_ids(packet)
    if not scope:
        return 1
    edges = packet["current_map"]["edges"]
    grew = True
    while grew:
        grew = False
        for edge in edges:
            if edge.get("type") == "composition" and edge["source_id"] in scope and edge["target_id"] not in scope:
                scope.add(edge["target_id"])
                grew = True
    by_id = {edge["id"]: edge for edge in edges}
    for update in answer["proposed_updates"]:
        content = update["content"]
        if any(node["id"] not in scope for node in content.get("replace_nodes", [])):
            return 0
        for edge_id in [edge["id"] for edge in content.get("replace_edges", [])] + content.get("remove_edge_ids", []):
            edge = by_id[edge_id]
            if not scope & {edge_id, edge["source_id"], edge["target_id"]}:
                return 0
    return 1


def local(card, text, finished=True):
    """L0, D1 and L2 for one generation."""
    answer = extract(text) if finished else None
    if not finished:
        errors = ["unfinished"]
    elif answer is None:
        errors = ["no_json_answer"]
    else:
        try:
            errors = validate_answer(answer, card["packet"])
        except Exception as error:  # malformed answers the validator does not anticipate
            errors = ["validator_exception:" + type(error).__name__]
    row = {"task_id": card["task_id"], "d1_card": bool(card["d1"]), "label": card["d1"], "valid": not errors,
           "errors": errors[:3], "answer": answer, "d1": None, "l2": None, "l3": None}
    if card["d1"]:  # as L1: an unparsable answer is wrong; schema errors gate the reward, not D1
        row["pred"] = predict(card["task_family"], card["question"], answer)
        row["d1"] = int(row["pred"] == card["d1"])
    if not errors:
        row["l2"] = l2(card["packet"], answer)
    return row


def total(row):
    if not row["valid"]:
        return 0.0
    if row["d1_card"]:
        return (2 * row["d1"] + row["l2"] + row["l3"]) / 4
    return (row["l2"] + row["l3"]) / 2


def score(cards, texts, finished, client, workers=64):
    """Rows with every component and the reward; L3 calls run in parallel."""
    rows = [local(c, t, f) for c, t, f in zip(cards, texts, finished)]

    def call(i):
        try:
            return judge(client, JUDGE, cards[i]["packet"], rows[i]["answer"], cards[i]["bundle"])
        except Exception as error:  # a judge failure must not stop training
            return {"verdicts": None, "error": repr(error)[:200]}

    valid = [i for i, r in enumerate(rows) if r["valid"]]
    with ThreadPoolExecutor(max(1, min(workers, len(valid)))) as pool:
        for i, result in zip(valid, pool.map(call, valid)):
            rows[i]["l3"] = result.get("l3") if result.get("verdicts") else None
            rows[i]["l3_failed"] = rows[i]["l3"] is None
            rows[i]["verdicts"] = result.get("verdicts")
    groups = defaultdict(list)
    for r in rows:
        if r["l3"] is not None:
            groups[r["task_id"]].append(r["l3"])
    for i in valid:
        if rows[i]["l3"] is None:
            rows[i]["l3"] = st.mean(groups[rows[i]["task_id"]]) if groups[rows[i]["task_id"]] else 0.5
    for r in rows:
        r["reward"] = total(r)
    return rows


class Reward:
    """TRL reward function. The dataset's `card` column holds each card as a JSON string."""

    __name__ = "hypomap"

    def __init__(self, log_path, eos_ids, workers=64):
        from openai import OpenAI
        self.client = OpenAI(max_retries=4, timeout=300)
        self.log_path = Path(log_path)
        self.eos_ids = set(eos_ids)
        self.workers = workers

    def __call__(self, prompts, completions, completion_ids, card, trainer_state=None, **kwargs):
        cards = [json.loads(c) for c in card]
        finished = [bool(ids) and ids[-1] in self.eos_ids for ids in completion_ids]
        rows = score(cards, completions, finished, self.client, self.workers)
        step = trainer_state.global_step if trainer_state is not None else None
        with self.log_path.open("a", encoding="utf-8") as sink:
            for r, text, ids in zip(rows, completions, completion_ids):
                sink.write(json.dumps({"step": step, **{k: v for k, v in r.items() if k not in ("answer", "verdicts")},
                                       "verdicts": r.get("verdicts"), "tokens": len(ids), "completion": text},
                                      ensure_ascii=False) + "\n")
        print(json.dumps({"step": step, **summary(rows)}), flush=True)
        return [r["reward"] for r in rows]


def summary(rows):
    mean = lambda xs: round(st.mean(xs), 4) if xs else None  # noqa: E731
    d1 = [r for r in rows if r["d1_card"]]
    return {"n": len(rows), "reward": mean([r["reward"] for r in rows]),
            "valid": mean([r["valid"] for r in rows]),
            "d1_accuracy": mean([r["d1"] for r in d1]),
            "d1_balanced_accuracy": balanced([{"d1": r["label"], "pred": r["pred"]} for r in d1]) if d1 else None,
            "l2": mean([r["l2"] for r in rows if r["valid"]]),
            "l3": mean([r["l3"] for r in rows if r["valid"]]),
            "l3_failed": sum(r.get("l3_failed", False) for r in rows),
            "reward_d1_cards": mean([r["reward"] for r in d1]),
            "reward_other_cards": mean([r["reward"] for r in rows if not r["d1_card"]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="inference.py output files")
    parser.add_argument("--cards", type=Path, required=True, help="cards.jsonl from package.py")
    parser.add_argument("--output", type=Path, help="Write per-model summaries here (JSON).")
    parser.add_argument("--rows", type=Path, help="Append per-answer rows here (JSONL).")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env", override=False)
    client = OpenAI(max_retries=4, timeout=300)
    cards = {}
    with args.cards.open(encoding="utf-8") as handle:
        for line in handle:
            card = json.loads(line)["card"]
            cards[card["task_id"]] = json.loads(card) if isinstance(card, str) else card
    by_model = defaultdict(list)
    for path in args.results:
        with path.open(encoding="utf-8") as handle:
            results = [json.loads(line) for line in handle if line.strip()]
        rows = score([cards[r["card_id"]] for r in results],
                     [r.get("raw_output") or "" for r in results],
                     [r.get("finish_reason") == "stop" for r in results], client, args.workers)
        model = results[0]["requested_model"]
        by_model[model] += rows
        if args.rows:
            with args.rows.open("a", encoding="utf-8") as sink:
                for r in rows:
                    sink.write(json.dumps({"model": model, "file": path.name,
                                           **{k: v for k, v in r.items() if k != "answer"}}, ensure_ascii=False) + "\n")
        print(path.name, json.dumps(summary(rows)), flush=True)
    report = {model: summary(rows) for model, rows in by_model.items()}
    columns = [("reward", "reward"), ("valid", "valid"), ("d1_balanced_accuracy", "D1 bal"), ("l2", "L2"),
               ("l3", "L3"), ("reward_d1_cards", "D1 cards"), ("reward_other_cards", "other")]
    print(f"\n{'model':16} {'n':>5} " + " ".join(f"{label:>8}" for _, label in columns))
    for model, s in sorted(report.items(), key=lambda kv: -(kv[1]["reward"] or 0)):
        print(f"{model:16} {s['n']:>5} " + " ".join(f"{'-' if s[k] is None else round(s[k], 3):>8}" for k, _ in columns))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
