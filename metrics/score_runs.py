#!/usr/bin/env python3
"""L0–L4 for generated answers on the split v2 development or test cards (docs/specs/metrics.md §2–7, §9).

    .venv/bin/python metrics/score_runs.py RESULTS.jsonl ... --split test --out DIR [--l3-judges gemini-3.8-flash luna]

Every layer scores every card; a failed L0 check does not set later layers to 0 (decided 2026-10-10, spec §2).
An answer is the final text after the thinking; a reply cut off while thinking, or with no text after it, has none.
- L0 (spec §3, relaxed 2026-10-06): complete (generation stopped before the budget), parse (a JSON answer follows
  the thinking; if the text is not bare JSON, the last fenced block, then the outermost brace span), schema_valid
  (`validate_answer` on the normalized answer, apart from the patch and the forgiven errors), patch_applies (on the
  normalized answer); valid = all four. strict_valid, the rule before 2026-10-06, is reported as a diagnostic.
- L1 (D1 cards): the named-concern rule of metrics/score_d1.py on the normalized JSON answer, whatever its
  L0; no JSON answer (or a JSON object with none of the answer's fields) is wrong. Balanced accuracy, act recall,
  over-edit rate (acting on a no-act card), pair accuracy (parent and variant both right).
- L2 (diagnostic): the §10 rule (metrics/reward.py `l2`) on the normalized JSON answer.
- L3: metrics/l3_judge.py, one call per answer and judge; the first of --l3-judges is the reported L3, the others
  are columns. The judge reads the JSON answer as written, or the final text when it is not JSON; no answer
  scores 0 without a call. Calls are appended to DIR/l3-<judge>.jsonl and reused on a rerun.
- L4: the workspace after the answer's patch (metrics/render_workspace.py); when the patch does not apply, or
  there is no answer, the workspace is unchanged (pre-patch). DIR/l4-workspaces.jsonl holds every distinct
  workspace of a card (pre-patch, reference-patched, each model's), identical workspaces once; the judges'
  scores go to DIR/l4-luna.jsonl and DIR/l4-qwen3-8b.jsonl. Card score: the mean of the four dimensions; the
  panel is the mean of the judges present.
Per task family, over all cards; 95% bootstrap intervals resampling papers (2,000 resamples); paired
differences against --baseline on the same cards.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "metrics")]

from hypomap.grading import apply_proposed_updates, validate_answer  # noqa: E402
from hypomap.tasks import OUTPUT_FIELDS, object_ids  # noqa: E402
from l3_judge import EXPORTS, judge, load_cards, openrouter, read_jsonl  # noqa: E402
from l4_judge import DIMENSIONS  # noqa: E402
from render_workspace import merged, render  # noqa: E402
from reward import JUDGE, extract, l2  # noqa: E402
from score_d1 import predict  # noqa: E402

RESAMPLES = 2000
JUDGES = ("luna", "qwen3-8b")  # the L4 panel (spec §7)
# L3 judges (spec §6): name -> (model, OpenRouter provider or None for the OpenAI Responses API)
L3_JUDGES = {"gemini-3.8-flash": ("google/gemini-3.8-flash", "google-ai-studio/flex"), "luna": (JUDGE, None)}
FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
# Relaxed L0 (spec §3, 2026-10-06). Missing text fields, availability and basis_ids of answer items are filled
# with these empty values; missing target or evidence IDs are not invented.
FILL = {"aspect_assessments": {"availability": "", "assessment": "", "scope": "", "limitations": [], "rationale": ""},
        "findings": {"kind": "", "description": "", "basis_ids": []},
        "verification_needs": {"unresolved_question": "", "needed_observations": [], "comparison_or_control": None,
                               "why_informative": "", "missing_design_information": []},
        "proposed_updates": {"basis_ids": [], "rationale": ""}}
PROVENANCE = ("source_spans", "source_ref", "provenance", "evidence_basis", "inference_reason")
# validate_answer errors the relaxed L0 forgives; references are rechecked by relaxed_errors (nonexistent IDs fail).
FORGIVEN = {"unknown_evidence_id", "invalid_assessment_targets", "unknown_or_invalid_reference", "unknown_preserved_id",
            "duplicate_evidence_id", "availability_not_publicly_defined", "missing_no_change_reason"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def read_answer(text):
    """The final answer after the thinking (reward.extract); if that is not bare JSON, the last fenced block,
    then the outermost brace span (the 2026-09-21 extraction). Fisher-R1 writes its reasoning as visible text
    and ends with a fenced JSON block; of the other test runs it changes only three BioMistral answers (none valid)."""
    answer = extract(text)
    if isinstance(answer, dict) or ("<think>" in text and "</think>" not in text):  # parsed, or cut off while thinking
        return answer
    text = text.rsplit("</think>", 1)[-1]
    for candidate in FENCE.findall(text)[-1:] + [text[text.find("{"):text.rfind("}") + 1]]:
        try:
            answer = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(answer, dict):
            return answer
    return None


def normalize(answer, packet):
    """A copy of the answer with the slips spec §3 forgives repaired: missing top-level collections empty, extra
    top-level fields dropped, missing item fields filled empty, and in replace_nodes an omitted type and every
    provenance field taken from the existing node."""
    a = {f: deepcopy(answer.get(f)) for f in OUTPUT_FIELDS}
    for f in OUTPUT_FIELDS[:-1]:
        if a[f] is None:
            a[f] = []
    for field, empty in FILL.items():
        for item in a[field] if isinstance(a[field], list) else []:
            if isinstance(item, dict):
                for k, v in empty.items():
                    item.setdefault(k, deepcopy(v))
    nodes = {n["id"]: n for n in packet["current_map"]["nodes"]}
    for update in a["proposed_updates"] if isinstance(a["proposed_updates"], list) else []:
        content = update.get("content") if isinstance(update, dict) else None
        replaced = content.get("replace_nodes") if isinstance(content, dict) else None
        for item in replaced if isinstance(replaced, list) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] in nodes:
                existing = nodes[item["id"]]
                item.setdefault("type", existing.get("type"))
                for k in PROVENANCE:
                    if k in existing:
                        item[k] = deepcopy(existing[k])
                    else:
                        item.pop(k, None)
    return a


def relaxed_errors(answer, packet):
    """validate_answer errors on the normalized answer apart from the patch and the forgiven ones (spec §3): any ID
    of the packet or of a node the answer adds may stand in any ID field; an ID that exists in neither stays an error."""
    errors = [e for e in validate_answer(answer, packet)
              if not e.startswith("unsafe_or_invalid_patch") and e.split(":")[0] not in FORGIVEN]
    if not errors:  # the collections are lists here
        known = object_ids(packet)
        for update in answer["proposed_updates"]:
            content = update.get("content") if isinstance(update, dict) else None
            added = content.get("add_nodes") if isinstance(content, dict) else None
            known |= {x["id"] for x in added or [] if isinstance(x, dict) and isinstance(x.get("id"), str)}
        fields = [(s, ("target_ids", "evidence_ids")) for s in answer["aspect_assessments"]]
        fields += [(item, ("target_ids", "basis_ids")) for f in ("findings", "proposed_updates", "verification_needs")
                   for item in answer[f]]
        ids = list(answer["preserved_ids"])
        for item, keys in fields:
            for k in keys:
                if isinstance(item, dict) and k in item:
                    ids += item[k] if isinstance(item[k], list) else [None]  # an ID field that is not a list fails
        if any(not isinstance(x, str) or x not in known for x in ids):
            errors.append("nonexistent_id")
    return errors


def final_text(raw):
    """The text after the thinking; None when the reply was cut off while thinking or has no text after it."""
    if "<think>" in raw and "</think>" not in raw:
        return None
    text = raw.rsplit("</think>", 1)[-1].strip()
    return text or None


def local(card, result, hide_merged=False):
    """L0, L1, L2 and the L4 workspace for one answer; hide_merged: leave out nodes a consolidation merged away."""
    raw = result.get("raw_output") or ""
    complete = result.get("finish_reason") in ("stop", "completed")  # vLLM, OpenAI Responses (Luna)
    text = final_text(raw)
    answer = read_answer(raw) if text else None
    parse = isinstance(answer, dict)
    is_answer = parse and bool(set(answer) & set(OUTPUT_FIELDS))  # an object with none of the fields is not an answer
    packet = card["packet"]
    schema = patch = strict = False
    normalized, patched, hide = None, packet, set()
    if parse:
        try:  # the rule before 2026-10-06, a diagnostic
            strict = not validate_answer(answer, packet)
            apply_proposed_updates(packet, answer["proposed_updates"])
        except Exception:
            strict = False
        try:
            normalized = normalize(answer, packet)
            schema = is_answer and not relaxed_errors(normalized, packet)
        except Exception:  # malformed answers the validator does not anticipate
            schema = False
        try:
            patched = apply_proposed_updates(packet, normalized["proposed_updates"])
            patch = True
            hide = merged(patched, normalized["proposed_updates"]) if hide_merged else set()
        except Exception:
            patched = packet  # the patch does not apply: the workspace is unchanged
    valid = complete and parse and schema and patch
    row = {"task_id": card["task_id"], "family": card["task_family"], "paper": card["paper_id"],
           "label": card["d1"], "is_variant": card["is_variant"], "parent": card["parent_task_id"],
           "answered": text is not None, "complete": complete, "parse": parse, "schema_valid": schema,
           "patch_applies": patch, "valid": valid, "strict_valid": complete and strict,
           "pred": None, "d1": None, "l2": None,
           "answer": answer if parse else text, "workspace": render(patched, hide)}
    if is_answer and normalized is not None:
        try:
            row["l2"] = l2(packet, normalized)
        except Exception:
            pass
    if card["d1"]:
        row["pred"] = predict(card["task_family"], card["question"], normalized) if is_answer and normalized else None
        row["d1"] = int(row["pred"] == card["d1"])
    return row


def l1(rows):
    act = [r for r in rows if r["label"] == "act"]
    no = [r for r in rows if r["label"] == "no-act"]
    by_id = {r["task_id"]: r for r in rows}
    pairs = [(r, by_id[r["parent"]]) for r in rows if r["is_variant"] and r["parent"] in by_id]
    recall = mean([r["d1"] for r in act])
    no_recall = mean([r["d1"] for r in no])
    return {"d1_balanced_accuracy": mean([recall, no_recall]), "act_recall": recall,
            "over_edit_rate": mean([int(r["pred"] == "act") for r in no]),
            "pair_accuracy": mean([a["d1"] * b["d1"] for a, b in pairs])}


def metrics(rows):
    out = {k: mean([float(r[k]) for r in rows]) for k in ("answered", "valid", "strict_valid", "complete", "parse",
                                                     "schema_valid", "patch_applies")}
    out["l2"] = mean([r["l2"] for r in rows])  # diagnostic, over JSON answers
    for k in [k for k in rows[0] if k.startswith("l3") and k != "l3_calls"] + ["coverage", "pitfall_rate"]:
        out[k] = mean([r.get(k) for r in rows])
    if any("l4_judges" in r for r in rows):
        for k in ["l4"] + ["l4_" + j for j in JUDGES] + ["l4_" + d for d in DIMENSIONS]:
            out[k] = mean([r.get(k) for r in rows])
    if any(r["label"] for r in rows):
        out.update(l1([r for r in rows if r["label"]]))
    return out


def bootstrap(by_model, baseline, seed=0):
    """Point values, 95% intervals and paired differences, resampling papers."""
    papers = sorted({r["paper"] for rows in by_model.values() for r in rows})
    index = {m: defaultdict(list) for m in by_model}
    for m, rows in by_model.items():
        for r in rows:
            index[m][r["paper"]].append(r)
    point = {m: metrics(rows) for m, rows in by_model.items()}
    draws = {m: defaultdict(list) for m in by_model}
    diffs = {m: defaultdict(list) for m in by_model if m != baseline}
    rng = random.Random(seed)
    for _ in range(RESAMPLES):
        sample = [rng.choice(papers) for _ in papers]
        values = {}
        for m in by_model:
            rows = [r for p in sample for r in index[m][p]]
            values[m] = metrics(rows) if rows else {}
        for m, v in values.items():
            for k, x in v.items():
                if x is not None:
                    draws[m][k].append(x)
                    if m in diffs and values.get(baseline, {}).get(k) is not None:
                        diffs[m][k].append(x - values[baseline][k])

    def interval(xs):
        xs = sorted(xs)
        return [round(xs[int(.025 * len(xs))], 4), round(xs[int(.975 * len(xs)) - 1], 4)] if len(xs) > 40 else None

    report = {}
    for m in by_model:
        report[m] = {k: {"value": None if v is None else round(v, 4), "ci": interval(draws[m][k])}
                     for k, v in point[m].items()}
        if m in diffs:
            for k, v in point[m].items():
                b = point.get(baseline, {}).get(k)
                if v is not None and b is not None:
                    report[m][k]["minus_" + baseline] = {"value": round(v - b, 4), "ci": interval(diffs[m][k])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="inference.py output files, one sample per card")
    parser.add_argument("--split", default="test", choices=("development", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l3-judges", nargs="*", default=["gemini-3.8-flash"], choices=list(L3_JUDGES),
                        help="The first is the reported L3; none skips L3.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--baseline", default="sft-150")
    parser.add_argument("--hide-merged", action="store_true", help="L4 workspaces leave out nodes merged away by a "
                        "consolidation (side-by-side judge, data/results/map-quality); off for the absolute L4.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bundles = load_cards() if args.l3_judges else {}  # the answer bundles are needed only by the L3 judges
    cards = {}
    for row in read_jsonl(EXPORTS / f"{args.split}.jsonl.gz"):
        meta = row["split_meta"]
        cards[meta["task_id"]] = {**meta, "packet": row["input"], "question": row["input"]["question"],
                                  "reference": row["output"], "bundle": bundles.get(meta["task_id"], {}).get("bundle")}

    by_model = {}
    for path in args.results:
        results = read_jsonl(path)
        model = results[0]["requested_model"]
        assert len({r["card_id"] for r in results}) == len(results) == len(cards), f"{path}: not one answer per card"
        by_model[model] = [local(cards[r["card_id"]], r, args.hide_merged) for r in results]

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    for name in args.l3_judges:  # one call per answer and judge; no answer scores 0 without a call
        model, provider = L3_JUDGES[name]
        calls_path = args.out / f"l3-{name}.jsonl"
        done = {}
        if calls_path.exists():
            done = {(c["model"], c["task_id"]): c for c in read_jsonl(calls_path) if c.get("verdicts")}
        jobs = [(m, r) for m, rows in by_model.items() for r in rows if r["answered"] and (m, r["task_id"]) not in done]
        if jobs:
            if provider:
                client, extra = openrouter(provider)
            else:
                from openai import OpenAI
                client, extra = OpenAI(max_retries=4, timeout=300), None

            def call(job):
                m, row = job
                card = cards[row["task_id"]]
                try:
                    result = judge(client, model, card["packet"], row["answer"], card["bundle"], extra=extra)
                except Exception as error:
                    result = {"verdicts": None, "error": repr(error)[:200]}
                return {"model": m, "task_id": row["task_id"], "judge": name, **result}

            print(f"L3 {name}: {len(jobs)} calls", flush=True)
            with calls_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(args.workers) as pool:
                for i, future in enumerate(as_completed([pool.submit(call, job) for job in jobs]), 1):
                    c = future.result()
                    if c["verdicts"] is None:  # retried on the next run
                        print("failed:", c["model"], c["task_id"], c.get("error", "unparsed"), flush=True)
                        continue
                    sink.write(json.dumps(c, ensure_ascii=False) + "\n")
                    sink.flush()
                    done[(c["model"], c["task_id"])] = c
                    if i % 200 == 0:
                        print(f"{i}/{len(jobs)}", flush=True)
        for m, rows in by_model.items():
            for r in rows:
                c = done.get((m, r["task_id"])) if r["answered"] else {"l3": 0.0, "coverage": 0.0, "pitfall_rate": None}
                r["l3_" + name] = c["l3"] if c else None  # None: the call is still missing
                if name == args.l3_judges[0]:
                    r.update({k: c[k] if c else None for k in ("l3", "coverage", "pitfall_rate")})
    l3_missing = {name: sum(1 for rows in by_model.values() for r in rows if r.get("l3_" + name) is None)
                  for name in args.l3_judges}

    # L4 workspaces: every distinct workspace of a card once; the ID follows the text, so it survives reruns.
    workspaces, fixed = {}, {}
    for rotation, task_id in enumerate(sorted(cards)):
        card = cards[task_id]
        updates = card["reference"].get("proposed_updates")
        reference = apply_proposed_updates(card["packet"], updates) if updates else card["packet"]
        texts = {"pre-patch": render(card["packet"]),
                 "reference": render(reference, merged(reference, updates) if args.hide_merged else ())}
        texts.update({m: r["workspace"] for m, rows in by_model.items() for r in rows if r["task_id"] == task_id})
        for name, text in texts.items():
            wid = f"{task_id}|{hashlib.sha1(text.encode()).hexdigest()[:12]}"
            ws = workspaces.setdefault(wid, {"workspace_id": wid, "group": card["paper_id"], "rotation": rotation,
                                             "task_id": task_id, "members": [], "text": text})
            ws["members"].append(name)
            if name in ("pre-patch", "reference"):
                fixed[(name, task_id)] = wid
    with (args.out / "l4-workspaces.jsonl").open("w", encoding="utf-8") as sink:
        for ws in workspaces.values():
            sink.write(json.dumps(ws, ensure_ascii=False) + "\n")
    l4 = defaultdict(dict)  # workspace_id -> judge -> {dimension: mean over samples}
    for j in JUDGES:
        if (args.out / f"l4-{j}.jsonl").exists():
            scored = defaultdict(list)
            for c in read_jsonl(args.out / f"l4-{j}.jsonl"):
                if c.get("parsed"):
                    scored[c["workspace_id"]].append(c["parsed"]["scores"])
            for wid, samples in scored.items():
                l4[wid][j] = {d: mean([s[d] for s in samples]) for d in DIMENSIONS}

    def l4_scores(wid):
        judged = l4.get(wid, {})
        out = {"l4_" + j: mean(list(judged[j].values())) if j in judged else None for j in JUDGES}
        out["l4"] = mean([out["l4_" + j] for j in JUDGES])
        out.update({"l4_" + d: mean([judged[j][d] for j in judged]) for d in DIMENSIONS})
        return out

    if l4:
        for m, rows in by_model.items():
            for r in rows:
                wid = f"{r['task_id']}|{hashlib.sha1(r['workspace'].encode()).hexdigest()[:12]}"
                r.update(l4_scores(wid))
                r["l4_judges"] = len(l4.get(wid, {}))

    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as sink:
        for m, rows in by_model.items():
            for r in rows:
                sink.write(json.dumps({"model": m, **{k: v for k, v in r.items() if k not in ("answer", "workspace")}},
                                      ensure_ascii=False) + "\n")
    families = sorted({r["family"] for rows in by_model.values() for r in rows})
    report = {"split": args.split, "cards": len(cards), "papers": len({c["paper_id"] for c in cards.values()}),
              "l3_judges": {name: L3_JUDGES[name] for name in args.l3_judges}, "l3_missing": l3_missing,
              "l4_workspaces": len(workspaces), "l4_judged": {j: sum(1 for w in l4.values() if j in w) for j in JUDGES},
              "l4_missing": {j: sum(1 for rows in by_model.values() for r in rows if r.get("l4_" + j) is None)
                             for j in JUDGES} if l4 else None,
              "l4_fixed": {f: {name: {k: mean([l4_scores(fixed[(name, t)])[k] for t, c in cards.items()
                                               if c["task_family"] == f]) for k in ("l4", "l4_luna", "l4_qwen3-8b")}
                               for name in ("pre-patch", "reference")}
                           for f in sorted({c["task_family"] for c in cards.values()})} if l4 else None,
              "resamples": RESAMPLES, "baseline": args.baseline,
              "d1_cards": bootstrap({m: [r for r in rows if r["label"]] for m, rows in by_model.items()}, args.baseline),
              "families": {f: bootstrap({m: [r for r in rows if r["family"] == f] for m, rows in by_model.items()},
                                        args.baseline) for f in families}}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    models = list(by_model)
    header = "".join(f"{m[:14]:>15}" for m in models)
    cell = lambda group, m, k: f"{str(group[m].get(k, {}).get('value', '-')):>15}"  # noqa: E731
    first = by_model[models[0]]
    label = f"D1 cards (n={sum(1 for r in first if r['label'])})"
    print(f"\n{label:26}{header}")
    for k in ("d1_balanced_accuracy", "act_recall", "over_edit_rate", "pair_accuracy", "answered", "valid"):
        print(f"  {k:24}" + "".join(cell(report["d1_cards"], m, k) for m in models))
    for k in ("answered", "valid", "strict_valid", "l2", "l3", "coverage", "l4"):
        if k in report["families"][families[0]][models[0]]:
            print(f"\n{k + ' by family':31}{header}")
            for f in families:
                n = sum(1 for r in first if r["family"] == f)
                print(f"  {f[:24]:24} {n:>4}" + "".join(cell(report["families"][f], m, k) for m in models))
    print(f"\nL3 calls missing: {l3_missing}; L4 scores missing: {report['l4_missing']}")
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'rows.jsonl'}")


if __name__ == "__main__":
    main()
