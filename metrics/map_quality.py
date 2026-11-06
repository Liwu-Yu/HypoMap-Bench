"""Map quality (paper §3.3): Gemini 3.8 Flash compares the updated map with the original, side by side.

The judge sees two maps as Workspace A and B and says, for each of the four dimensions of the expert rubric
(metrics/l4_judge.py DIMENSIONS), which is better or that they tie; every pair in both orders, no examples.
Maps: data/results/map-quality/workspaces/l4-workspaces.jsonl, written by
`metrics/score_runs.py ... --hide-merged --l3-judges --out data/results/map-quality/workspaces` (nodes a
consolidation merged away are left out). Calls are appended to DIR/calls-<judge>.jsonl and reused.

    python metrics/map_quality.py run --judge gemini --models   # pairs not yet in DIR/calls-gemini.jsonl
    python metrics/map_quality.py score --judge gemini          # DIR/scores-gemini.json, DIR/task-gains-gemini.jsonl
    python metrics/map_quality.py report                        # reference update vs original
    python metrics/map_quality.py alignment                     # expert alignment (method only, see below)

Expert alignment (paper §3.3): the judge compares all 190 pairs of the 20 expert-rated maps of a separate
planning study in both orders; a map scores its mean preference over its 19 pairs, and metrics/l4_calibrate.py
correlates these scores with the expert ratings. The maps (participants' research plans) and their ratings are
not released, so `run` and `alignment` cover the expert set only where EXPERT_MAPS and EXPERT_RATINGS exist.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data/results/map-quality"  # --dir
EXPERT_MAPS = ROOT / "data/expert-alignment/maps.jsonl"  # not released
EXPERT_RATINGS = ROOT / "data/expert-alignment/ratings.jsonl"  # not released
sys.path.insert(0, str(ROOT / "metrics"))
from l4_judge import DIMENSIONS, first_json  # noqa: E402

JUDGES = {"luna": ("gpt-6-luna", None), "gemini": ("google/gemini-3.8-flash", "google-ai-studio/flex")}
WORKSPACES = HERE / "workspaces/l4-workspaces.jsonl"

SYSTEM = (
    "You are a blinded expert reviewer of scientific hypothesis-validation workspaces. You are shown two "
    "workspaces, A and B. For each dimension, decide which workspace better meets the question and criteria below, "
    "exactly as an expert judge would, or answer tie when neither is better. Judge only the workspaces shown; you "
    "are not told who or what produced them.\n\n"
    + "\n".join(f"- {key} ({name}): {criteria}" for key, (name, criteria) in DIMENSIONS.items())
    + "\n\nEach workspace is a hypothesis-validation plan written before any experiment is run, so its hypotheses "
      "are untested by design. Judge how well each plan states, organizes, links and designs the validation, not "
      "whether the hypotheses are true or already established by the cited literature. The two workspaces may "
      "plan different studies, or be two versions of one plan that differ in a few places; then compare the places "
      "where they differ. Compare what the workspaces contain, not their length, and do not prefer A or B for its "
      "position."
    + "\n\nReturn only a JSON object: {\"preferences\": {dimension: \"A\" | \"B\" | \"tie\"}, "
      "\"reasons\": {dimension: one sentence}} with the keys " + ", ".join(DIMENSIONS) + "."
)


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def pairs(models=False):
    """(set, pair_id, first, second) and texts by id; first/second keep a fixed meaning across both orders.
    models: also every model map that differs from its original (the full run)."""
    texts, out = {}, []
    for w in read_jsonl(EXPERT_MAPS) if EXPERT_MAPS.exists() else []:
        texts[w["workspace_id"]] = w["text"]
    out += [("expert", f"{a}|{b}", a, b) for a, b in itertools.combinations(sorted(texts), 2)]
    member = {}
    for w in read_jsonl(WORKSPACES):
        texts[w["workspace_id"]] = w["text"]
        member.update({(m, w["task_id"]): w["workspace_id"] for m in w["members"]})
    for task in sorted({t for _, t in member}):
        pre, ref = member[("pre-patch", task)], member[("reference", task)]
        if pre != ref:
            out.append(("reference", task, pre, ref))
    if models:  # each model map that differs from the original (and from the reference update) against the original
        seen = {(first, second) for _, _, first, second in out}
        for (m, task), w in sorted(member.items()):
            pre = member[("pre-patch", task)]
            if m not in ("pre-patch", "reference") and w != pre and (pre, w) not in seen:
                seen.add((pre, w))
                out.append(("model", f"{task}|{w.split('|')[1]}", pre, w))
    return out, texts


def parse(raw):
    obj = first_json(raw)
    try:
        prefs = {d: str(obj["preferences"][d]).strip().upper() for d in DIMENSIONS}
    except (KeyError, TypeError):
        return None
    prefs = {d: {"A": "A", "B": "B", "TIE": "tie"}.get(p) for d, p in prefs.items()}
    return None if None in prefs.values() else {"preferences": prefs, "reasons": obj.get("reasons", {})}


def load(judge):
    """Parsed calls of the current pairs: pair_id -> order -> call (a call on an older rendering is ignored)."""
    path = HERE / f"calls-{judge}.jsonl"
    current = {(pid, first, second) for _, pid, first, second in pairs(models=True)[0]}
    out = defaultdict(dict)
    for c in read_jsonl(path) if path.exists() else []:
        if c.get("parsed") and (c["pair_id"], c["first"], c["second"]) in current:
            out[c["pair_id"]][c["order"]] = c
    return out


def run(judge, limit, models=False):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    model, provider = JUDGES[judge]
    if provider:
        from l3_judge import openrouter
        client, extra = openrouter(provider)
    else:
        from openai import OpenAI
        client, extra = OpenAI(max_retries=4, timeout=900), None
    all_pairs, texts = pairs(models)
    done = load(judge)
    jobs = [(s, pid, first, second, order) for s, pid, first, second in all_pairs for order in (0, 1)
            if order not in done.get(pid, {})]
    jobs = jobs[:limit] if limit else jobs
    print(f"{judge}: {len(jobs)} calls", flush=True)

    def call(job):
        s, pid, first, second, order = job
        a, b = (first, second) if order == 0 else (second, first)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"## Workspace A\n{texts[a]}\n\n## Workspace B\n{texts[b]}"}]
        try:
            if provider:  # model defaults for sampling and reasoning, as for the correctness judge
                response = client.chat.completions.create(model=model, messages=messages, max_tokens=16384,
                                                          extra_body=extra)
                raw = response.choices[0].message.content or ""
            else:
                response = client.responses.create(model=model, input=messages, store=False, max_output_tokens=8192)
                raw = response.output_text
            usage = response.usage.model_dump() if response.usage else None
        except Exception as error:
            print(pid, order, "error", repr(error)[:200], flush=True)
            return None
        return {"set": s, "pair_id": pid, "order": order, "first": first, "second": second, "A": a, "B": b,
                "judge": judge, "parsed": parse(raw), "raw": raw, "usage": usage}

    with (HERE / f"calls-{judge}.jsonl").open("a", encoding="utf-8") as sink, ThreadPoolExecutor(16) as pool:
        for i, future in enumerate(as_completed([pool.submit(call, j) for j in jobs]), 1):
            row = future.result()
            if row is None:
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if not row["parsed"] or i % 50 == 0:
                print(i, row["pair_id"], row["order"], "ok" if row["parsed"] else "unparsed", flush=True)


def side(call, dim):
    """+1 when the pair's first map is preferred on dim, -1 the second, 0 tie."""
    p = call["parsed"]["preferences"][dim]
    return 0 if p == "tie" else (1 if call[p] == call["first"] else -1)


def tau_b(xs, ys):
    """Kendall tau-b over pairs given as signs: x = judge, y = experts."""
    c = sum(1 for x, y in zip(xs, ys) if x * y > 0)
    d = sum(1 for x, y in zip(xs, ys) if x * y < 0)
    n, tx, ty = len(xs), sum(1 for x in xs if x == 0), sum(1 for y in ys if y == 0)
    return (c - d) / math.sqrt((n - tx) * (n - ty))


def pair_scores(judges):
    """pair_id -> dim -> mean over judges and both orders of side(); pairs complete for every judge only."""
    calls = {j: load(j) for j in judges}
    ids = set.intersection(*({p for p, v in calls[j].items() if len(v) == 2} for j in judges))
    sets = {p: calls[judges[0]][p][0]["set"] for p in ids}
    firsts = {p: (calls[judges[0]][p][0]["first"], calls[judges[0]][p][0]["second"]) for p in ids}
    return {p: {d: mean(side(calls[j][p][o], d) for j in judges for o in (0, 1)) for d in DIMENSIONS}
            for p in ids}, sets, firsts


def report():
    sign = lambda x: (x > 1e-9) - (x < -1e-9)  # noqa: E731
    ratings = {r["artifact_id"]: r["scores"] for r in (read_jsonl(EXPERT_RATINGS) if EXPERT_RATINGS.exists() else [])}
    expert_mean = {a: {d: mean(s[e][d] for e in s) for d in DIMENSIONS} for a, s in ratings.items()}
    family = {r["task_id"]: r["family"] for r in read_jsonl(HERE / "workspaces/rows.jsonl")}
    out = {}
    available = [j for j in JUDGES if (HERE / f"calls-{j}.jsonl").exists()]
    for name, judges in [(j, [j]) for j in available] + ([("panel", available)] if len(available) > 1 else []):
        scores, sets, firsts = pair_scores(judges)
        res = {"pairs": len(scores)}
        if len(judges) == 1:
            calls = load(judges[0])
            usage = [c["usage"] for v in calls.values() for c in v.values() if c.get("usage")]
            res["order_agreement"] = round(mean(side(v[0], d) == side(v[1], d) for v in calls.values() if len(v) == 2
                                                for d in DIMENSIONS), 3)
            res["usage"] = {"calls": len(usage), "input": sum(u.get("input_tokens", u.get("prompt_tokens", 0)) for u in usage),
                            "output": sum(u.get("output_tokens", u.get("completion_tokens", 0)) for u in usage),
                            "cost": round(sum(u.get("cost") or 0 for u in usage), 2)}
        ex = [p for p in scores if sets[p] == "expert"]
        res["expert_pair_tau_b"] = {}
        for d in (list(DIMENSIONS) + ["overall"]) if ex else []:
            dims = list(DIMENSIONS) if d == "overall" else [d]
            js = [sign(mean(scores[p][k] for k in dims)) for p in ex]
            es = [sign(mean(expert_mean[firsts[p][0]][k] - expert_mean[firsts[p][1]][k] for k in dims)) for p in ex]
            res["expert_pair_tau_b"][d] = round(tau_b(js, es), 3)
        ref = {p: -mean(scores[p].values()) for p in scores if sets[p] == "reference"}  # + : reference better
        share = lambda xs: {"better": round(mean(x > 1e-9 for x in xs), 3),  # noqa: E731
                            "tie": round(mean(abs(x) <= 1e-9 for x in xs), 3),
                            "worse": round(mean(x < -1e-9 for x in xs), 3), "mean": round(mean(xs), 3), "n": len(xs)}
        res["reference"] = {"all": share(list(ref.values())),
                            "by_dimension": {k: round(mean(-scores[p][k] for p in ref), 3) for k in DIMENSIONS},
                            "by_family": {f: share([s for t, s in ref.items() if family[t] == f])
                                          for f in sorted({family[t] for t in ref})}}
        out[name] = res
    (HERE / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    for name, res in out.items():
        r = res["reference"]["all"]
        print(f"{name:7} pairs {res['pairs']}  expert tau-b overall {res['expert_pair_tau_b'].get('overall', float('nan')):.2f}  "
              f"reference better {r['better']:.0%} tie {r['tie']:.0%} worse {r['worse']:.0%}  "
              + (f"orders agree {res['order_agreement']:.0%}  usage {res['usage']}" if "usage" in res else ""))
        print("        by family (better/worse): " + "; ".join(
            f"{f[:14]} {v['better']:.0%}/{v['worse']:.0%}" for f, v in res["reference"]["by_family"].items()))


def alignment():
    """Per-map scores on the expert set (mean preference over a map's pairs), correlated with the experts by
    metrics/l4_calibrate.py exactly as the absolute judge was."""
    files = []
    for judge in [j for j in JUDGES if (HERE / f"calls-{j}.jsonl").exists()]:
        scores = defaultdict(lambda: defaultdict(list))
        for pid, v in load(judge).items():
            if len(v) < 2 or v[0]["set"] != "expert":
                continue
            for o in (0, 1):
                for d in DIMENSIONS:
                    s = side(v[o], d)
                    scores[v[o]["first"]][d].append(s)
                    scores[v[o]["second"]][d].append(-s)
        path = HERE / f"expert-scores-{judge}.jsonl"
        with path.open("w") as sink:
            for ws, dims in sorted(scores.items()):
                sink.write(json.dumps({"workspace_id": ws, "judge": judge, "sample": 0,
                                       "parsed": {"scores": {d: mean(x) for d, x in dims.items()}}}) + "\n")
        files.append(str(path))
    files = [f for f in files if Path(f).stat().st_size]
    if not files or not EXPERT_RATINGS.exists():
        sys.exit("No expert-set calls or ratings: the expert-rated maps and their ratings are not released.")
    subprocess.run([sys.executable, str(ROOT / "metrics/l4_calibrate.py"), *files, "--experts", str(EXPERT_RATINGS), "--output",
                    str(HERE / "alignment.json")], cwd=ROOT, check=True)


def score(judges):
    """Map gain per model and task: 0 when the map is unchanged (no answer, a patch that does not apply, or no
    edit); otherwise the mean over the judges, both orders and the four dimensions of +1 (the model's map is
    better than the original), 0 (tie) or -1. Pooled over all test tasks: mean gain, shares improved and
    worsened; 95% intervals resampling papers (2,000, seed 0), paired differences against BASELINES."""
    import random
    scores, sets, firsts = pair_scores(judges)
    by_pair = {firsts[p]: -mean(v.values()) for p, v in scores.items()}  # + : second (updated) map better
    by_dim = {firsts[p]: {d: -x for d, x in v.items()} for p, v in scores.items()}
    member = {(m, w["task_id"]): w["workspace_id"] for w in read_jsonl(WORKSPACES) for m in w["members"]}
    rows = read_jsonl(HERE / "workspaces/rows.jsonl")
    info = {r["task_id"]: r for r in rows}
    models = list(dict.fromkeys(r["model"] for r in rows)) + ["reference"]
    table, dims = defaultdict(dict), defaultdict(dict)  # model -> task -> gain; -> {dimension: gain}
    for m in models:
        for t in info:
            w, pre = member[(m, t)], member[("pre-patch", t)]
            table[m][t] = 0.0 if w == pre else by_pair.get((pre, w))
            dims[m][t] = {d: 0.0 for d in DIMENSIONS} if w == pre else by_dim.get((pre, w))
    missing = {m: sum(v is None for v in table[m].values()) for m in models}
    papers = sorted({r["paper"] for r in info.values()})
    tasks_of = defaultdict(list)
    for t, r in info.items():
        tasks_of[r["paper"]].append(t)

    def stats(m, tasks):
        xs = [table[m][t] for t in tasks if table[m][t] is not None]
        return {"gain": mean(xs), "improved": mean(x > 1e-9 for x in xs), "worsened": mean(x < -1e-9 for x in xs)}

    point = {m: stats(m, list(info)) for m in models}
    rng, draws = random.Random(0), defaultdict(list)
    for _ in range(2000):
        sample = [t for p in (rng.choice(papers) for _ in papers) for t in tasks_of[p]]
        draws[0].append({m: stats(m, sample) for m in models})
    ci = lambda xs: [round(sorted(xs)[50], 3), round(sorted(xs)[1949], 3)]  # noqa: E731
    out = {"judges": judges, "missing": missing, "models": {}}
    for m in models:
        out["models"][m] = {k: {"value": round(v, 3), "ci": ci([d[m][k] for d in draws[0]])} for k, v in point[m].items()}
        for b in BASELINES:
            if m != b and b in models:
                out["models"][m]["gain"][f"minus_{b}"] = {
                    "value": round(point[m]["gain"] - point[b]["gain"], 3),
                    "ci": ci([d[m]["gain"] - d[b]["gain"] for d in draws[0]])}
    families = sorted({r["family"] for r in info.values()})
    out["families"] = {f: {m: round(stats(m, [t for t in info if info[t]["family"] == f])["gain"], 3) for m in models}
                       for f in families}
    name = "-".join(judges)
    (HERE / f"scores-{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    with (HERE / f"task-gains-{name}.jsonl").open("w") as sink:
        for m in models:
            for t, g in table[m].items():
                sink.write(json.dumps({"model": m, "task_id": t, "family": info[t]["family"], "paper": info[t]["paper"],
                                       "gain": g, "dimensions": dims[m][t]}) + "\n")
    print(f"judges {judges}; missing {missing}")
    for m, v in out["models"].items():
        d = {b: v["gain"].get(f"minus_{b}") for b in BASELINES}
        print(f"{m.split('/')[-1]:22} gain {v['gain']['value']:+.3f} {v['gain']['ci']}  improved {v['improved']['value']:.2f}"
              f"  worsened {v['worsened']['value']:.2f}  " + "  ".join(
                  f"-{b.split('/')[-1]} {x['value']:+.3f} {x['ci']}" for b, x in d.items() if x))


BASELINES = ("sft-150", "Qwen/Qwen3-8B", "sft-v1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report", "alignment", "score", "prompt"))
    parser.add_argument("--judge", choices=list(JUDGES), default="gemini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--models", action="store_true", help="Full run: every model map that differs from its original.")
    parser.add_argument("--dir", type=Path, default=HERE, help="Folder with workspaces/ and the call files.")
    args = parser.parse_args()
    HERE, WORKSPACES = args.dir, args.dir / "workspaces/l4-workspaces.jsonl"
    if args.command == "run":
        run(args.judge, args.limit, args.models)
    elif args.command == "report":
        report()
    elif args.command == "alignment":
        alignment()
    elif args.command == "score":
        score([args.judge])
    else:
        print(SYSTEM)
