#!/usr/bin/env python3
"""L3 grader-accuracy and signal checks on development cards (docs/specs/metrics.md §6, §10).

    python metrics/l3_check.py build --out DIR      # answer sets; the drop set needs one Luna rewrite per card
    python metrics/l3_judge.py --answers DIR/answers.jsonl --output DIR/luna.jsonl --samples 2
    python metrics/l3_check.py report --out DIR [--second DIR/second.jsonl]
    python metrics/l3_check.py report --out DIR --judge qwen3-8b --second DIR/luna.jsonl   # Qwen, and the panel

Answer sets (ids "kind|task_id|item"):
- reference: the reference answer; requirements should be met, pitfalls absent (false rejection);
- plant: the reference answer with one forbidden inference appended to a text field; that pitfall
  should be caught (false acceptance);
- drop: the reference answer rewritten by Luna to stop meeting one requirement; that requirement
  should be judged unmet (false acceptance);
- step125: SFT v2 step-125's three development samples on the D1 cards; how much room L3 leaves and
  whether it differs between samples of one card more than between two judge calls (signal).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import random
import statistics as st

from l3_judge import ROOT, items, load_cards, read_jsonl

STEP125 = [ROOT / f"data/results/judge-checks/step-125{s}.jsonl" for s in ("", "-r2", "-r3")]
TEXT_KEYS = ("assessment", "description", "rationale", "no_change_reason", "statement")

REWRITE = (
    "You edit a JSON answer to a scientific workspace task for a grader test. Rewrite the answer minimally so "
    "that it no longer meets the given requirement: remove or alter only the content that meets it, keep "
    "everything else unchanged, and keep the same JSON structure. Return only the JSON answer."
)


def plant(answer, inference):
    """Append the forbidden inference, as an assertion, to the first substantial text field."""
    answer = copy.deepcopy(answer)
    sentence = inference.strip().rstrip(".") + "."

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in TEXT_KEYS and isinstance(value, str) and len(value) > 40:
                    node[key] = value.rstrip() + " " + sentence
                    return True
                if walk(value):
                    return True
        elif isinstance(node, list):
            return any(walk(value) for value in node)
        return False

    return answer if walk(answer) else None


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    cards = {t: c for t, c in load_cards().items() if c["split"] == "development"}
    rng = random.Random(0)
    rows, rewrites = [], []
    for task_id, card in sorted(cards.items()):
        ref, card_items = card["reference"], items(card["bundle"])
        rows.append({"id": f"reference|{task_id}|", "task_id": task_id, "answer": ref})
        pits = [i for i in card_items if i["type"] == "pitfall"]
        if pits:
            item = rng.choice(pits)
            planted = plant(ref, item["forbidden_inference"])
            if planted:
                rows.append({"id": f"plant|{task_id}|{item['id']}", "task_id": task_id, "answer": planted})
        reqs = [i for i in card_items if i["type"] == "requirement"]
        if reqs:
            rewrites.append((task_id, ref, rng.choice(reqs)))
    for path in STEP125:
        for row in read_jsonl(path):
            if row["card_id"] in cards and isinstance(row.get("parsed_answer"), dict):
                rows.append({"id": f"step125|{row['card_id']}|{path.stem}", "task_id": row["card_id"],
                             "answer": row["parsed_answer"]})

    from openai import OpenAI
    client = OpenAI()

    def rewrite(job):
        task_id, ref, item = job
        response = client.responses.create(model="gpt-6-luna", store=False, max_output_tokens=16384, input=[
            {"role": "system", "content": REWRITE},
            {"role": "user", "content": json.dumps({"requirement": item["requirement"], "answer": ref}, ensure_ascii=False)}])
        text = response.output_text or ""
        try:
            answer = json.JSONDecoder().raw_decode(text[text.find("{"):])[0]
        except ValueError:
            return None
        if isinstance(answer.get("answer"), dict) and "requirement" in answer:  # echoed the request
            answer = answer["answer"]
        return {"id": f"drop|{task_id}|{item['id']}", "task_id": task_id, "answer": answer} if answer != ref else None

    with ThreadPoolExecutor(16) as pool:
        rows += [r for r in pool.map(rewrite, rewrites) if r]
    with (out / "answers.jsonl").open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    kinds = [r["id"].split("|")[0] for r in rows]
    print({k: kinds.count(k) for k in dict.fromkeys(kinds)})


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(out, second, judge="luna"):
    calls = [r for r in read_jsonl(out / f"{judge}.jsonl")]
    parsed = [r for r in calls if r["verdicts"]]
    print(f"calls {len(calls)}, unparsed {len(calls) - len(parsed)}")
    by = {}
    for r in parsed:
        by.setdefault(r["id"], {})[r["sample"]] = r

    def verdicts(kind):
        return [(i, s, r["verdicts"]) for i, samples in by.items() if i.startswith(kind + "|") for s, r in samples.items()]

    ref = verdicts("reference")
    req = [v for _, _, vs in ref for k, v in vs.items() if k.startswith("R")]
    pit = [v for _, _, vs in ref for k, v in vs.items() if k.startswith("P")]
    print(f"reference answers: requirement met {sum(req)}/{len(req)} = {sum(req)/len(req):.3f}; "
          f"pitfall flagged {sum(pit)}/{len(pit)} = {sum(pit)/len(pit):.3f} (false rejection)")
    for kind, want in (("plant", True), ("drop", False)):
        rows = verdicts(kind)
        hit = [vs[i.split("|")[2]] == want for i, _, vs in rows]
        print(f"{kind}: target item caught {sum(hit)}/{len(hit)} = {sum(hit)/len(hit):.3f}")
        others = [v == (k.startswith("R")) for i, _, vs in rows for k, v in vs.items() if k != i.split("|")[2]]
        print(f"  other items unchanged from a correct reference: {sum(others)}/{len(others)} = {sum(others)/len(others):.3f}")

    pairs = [(s[0]["verdicts"], s[1]["verdicts"]) for s in by.values() if 0 in s and 1 in s]
    a = [x[k] for x, y in pairs for k in x]
    b = [y[k] for x, y in pairs for k in x]
    print(f"rerun agreement: {sum(x == y for x, y in zip(a, b))}/{len(a)} = {sum(x == y for x, y in zip(a, b))/len(a):.3f}, "
          f"kappa {kappa(a, b):.3f}")

    # Signal: step-125 L3 per card, between-sample spread vs two calls on the same answer.
    l3 = {}
    for i, samples in by.items():
        if i.startswith("step125|"):
            l3.setdefault(i.split("|")[1], {})[i] = st.mean(r["l3"] for r in samples.values())
    means = [st.mean(v.values()) for v in l3.values()]
    spread = [st.pstdev(v.values()) for v in l3.values() if len(v) > 1]
    rerun = [abs(s[0]["l3"] - s[1]["l3"]) / 2 for i, s in by.items() if i.startswith("step125|") and 0 in s and 1 in s]
    differ = sum(max(v.values()) - min(v.values()) > 0 for v in l3.values() if len(v) > 1)
    print(f"step-125: {len(l3)} cards, mean L3 {st.mean(means):.3f} (room {1 - st.mean(means):.3f}); "
          f"cards whose samples differ {differ}/{len(spread)}; between-sample SD {st.mean(spread):.3f} "
          f"vs rerun SD {st.mean(rerun):.3f}")
    cov = [r["coverage"] for i, s in by.items() if i.startswith("step125|") for r in s.values() if r["coverage"] is not None]
    pr = [r["pitfall_rate"] for i, s in by.items() if i.startswith("step125|") for r in s.values() if r["pitfall_rate"] is not None]
    print(f"  requirement coverage {st.mean(cov):.3f}, pitfall rate {st.mean(pr):.3f}")

    if second:
        other = {r["id"]: r["verdicts"] for r in read_jsonl(second) if r.get("verdicts")}
        common = [(by[i][0]["verdicts"], v) for i, v in other.items() if i in by and 0 in by[i]]
        a = [x[k] for x, y in common for k in x if k in y]
        b = [y[k] for x, y in common for k in x if k in y]
        print(f"second judge: {len(common)} answers, {len(a)} items, agreement "
              f"{sum(x == y for x, y in zip(a, b))/len(a):.3f}, kappa {kappa(a, b):.3f}")

        # Panel: each item's score (met requirement, absent pitfall = 1) averaged over the two judges' first calls.
        def score(i, vs):
            return {k: float(v == k.startswith("R")) for k, v in vs.items()}
        panel = {i: {k: (score(i, by[i][0]["verdicts"])[k] + s) / 2 for k, s in score(i, v).items()
                     if k in by[i][0]["verdicts"]} for i, v in other.items() if i in by and 0 in by[i]}
        ref = [s for i, p in panel.items() if i.startswith("reference|") for s in [p]]
        req = [v for p in ref for k, v in p.items() if k.startswith("R")]
        pit = [1 - v for p in ref for k, v in p.items() if k.startswith("P")]
        print(f"panel: reference requirement met {st.mean(req):.3f}, pitfall flagged {st.mean(pit):.3f}; "
              f"card L3 {st.mean(st.mean(p.values()) for p in ref):.3f}")
        for kind in ("plant", "drop"):
            hit = [1 - p[i.split("|")[2]] for i, p in panel.items() if i.startswith(kind + "|")]
            print(f"  {kind}: target item caught {st.mean(hit):.3f} of {len(hit)}")
        s125 = [st.mean(p.values()) for i, p in panel.items() if i.startswith("step125|")]
        print(f"  step-125: mean L3 {st.mean(s125):.3f} over {len(s125)} answers")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--judge", default="luna", help="the judge file DIR/JUDGE.jsonl to report")
    args = parser.parse_args()
    build(args.out) if args.command == "build" else report(args.out, args.second, args.judge)


if __name__ == "__main__":
    main()
