"""Paper tables and the family heatmap from this folder's scores (no-cascade metrics, Gemini L3) and the map quality of
data/results/map-quality (Gemini side by side with the original map; task-gains-gemini.jsonl).

Writes analysis/out/tables/{main-results,ci,paired-diffs,family-results,decision-family,pooled-extra}.tex and
analysis/out/family-gains.tex (the TikZ source of Figure 3; it needs the paper's figure preamble to compile).

    python analysis/pooled.py && python analysis/paper_tables.py
"""
import gzip
import random
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "data/results/final-seven"
TABLES = ROOT / "analysis/out/tables"
FIGURE = ROOT / "analysis/out/family-gains.tex"
TABLES.mkdir(parents=True, exist_ok=True)
SOURCE = "data/results/final-seven"

LUNA = "gpt-6-luna"
OPEN = ["BioMistral/BioMistral-7B", "May2222/Fisher-R1-14B", "Qwen/Qwen3-8B"]
OURS = ["sft-v1", "sft-150", "core-grpo-40"]
NAME = {LUNA: "GPT-6 Luna", "BioMistral/BioMistral-7B": "BioMistral-7B", "May2222/Fisher-R1-14B": "Fisher-R1-14B",
        "Qwen/Qwen3-8B": "Qwen3-8B", "sft-v1": "SFT w/o pairs", "sft-150": "SFT", "core-grpo-40": "SFT+RL"}
FAMILIES = [("initial_organization", "Initial organization", "Initial\\\\org."),
            ("focused_decomposition", "Focused decomposition", "Focused\\\\decomp."),
            ("required_aspect_completion", "Aspect completion", "Aspect\\\\compl."),
            ("verification_gap", "Verification gap", "Verif.\\\\gap"),
            ("evidence_audit", "Evidence audit", "Evidence\\\\audit"),
            ("guided_preservation", "Guided change", "Guided\\\\change"),
            ("local_repair", "Local repair$^\\dagger$", "Local\\\\repair$^\\dagger$"),
            ("scoped_alternative_explanation", "Alternative expl.$^\\dagger$", "Alt.\\\\expl.$^\\dagger$"),
            ("redundancy_consolidation", "Redundancy$^\\dagger$", "Redun-\\\\dancy$^\\dagger$")]
DIMS = ["mq_hypothesis_clarity", "mq_hierarchical_coherence", "mq_traceability", "mq_testability"]

read = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
rows = read(HERE / "rows.jsonl")
pooled = json.loads((HERE / "pooled.json").read_text())
summary = json.loads((HERE / "summary.json").read_text())
by_model = defaultdict(list)
for r in rows:
    by_model[r["model"]].append(r)


def fmt(v, kind="p"):
    if v is None:
        return "--"
    if kind == "q":
        return f"{v:.2f}"
    if kind == "g":  # map quality, -1 to 1
        s = f"{abs(v):.2f}".lstrip("0")
        return ("$-$" if round(v, 2) < 0 else "") + s
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s


def marks(values, lower=False):
    """Bold best, underline second best among the values given (by the printed value); ties share a mark."""
    shown = sorted({v for v in values.values() if v is not None}, reverse=not lower)
    best, second = (shown + [None, None])[:2]
    return {m: ("\\textbf{%s}" if v == best else "\\second{%s}" if v == second else "%s") if v is not None else "%s"
            for m, v in values.items()}


def cells(models, getter, kind="p", lower=False, rank=None):
    """Printed cells for models; rank: the models among which best and second best are marked."""
    vals = {m: getter(m) for m in models}
    shown = {m: None if v is None else round(v, 2) for m, v in vals.items()}
    style = marks({m: shown[m] for m in (rank or models)}, lower)
    return [style.get(m, "%s") % fmt(vals[m], kind) for m in models]


# Map quality: the side-by-side gain over the original map (data/results/map-quality), pooled with 95% intervals
# resampling papers (2,000, seed 0), per family, and SFT+RL minus SFT per family
REF = "reference"
gains = defaultdict(dict)
for r in read(ROOT / "data/results/map-quality/task-gains-gemini.jsonl"):
    gains[r["model"]][r["task_id"]] = r
MQ = ["mq"] + DIMS + ["improved", "worsened"]


def mq_values(r):
    g = r["gain"]
    return [g] + [r["dimensions"][d.removeprefix("mq_")] for d in DIMS] + [float(g > 1e-9), float(g < -1e-9)]


papers = sorted({r["paper"] for r in by_model["sft-150"]})
agg = {m: {p: [0] + [0.0] * len(MQ) for p in papers} for m in gains}  # per paper: count, sums
fam_agg = defaultdict(lambda: [0, 0.0])  # (family, paper, model) -> count, gain sum
for m, tasks in gains.items():
    for r in tasks.values():
        a = agg[m][r["paper"]]
        a[0] += 1
        for i, v in enumerate(mq_values(r), 1):
            a[i] += v
        fa = fam_agg[(r["family"], r["paper"], m)]
        fa[0] += 1
        fa[1] += r["gain"]


def mq_point(m, sample):
    n = sum(agg[m][p][0] for p in sample)
    return {k: sum(agg[m][p][i] for p in sample) / n for i, k in enumerate(MQ, 1)}


def fam_point(f, m, sample):
    n = sum(fam_agg[(f, p, m)][0] for p in sample)
    return sum(fam_agg[(f, p, m)][1] for p in sample) / n if n else None


mq = {m: {k: {"value": v} for k, v in mq_point(m, papers).items()} for m in gains}
mq_family = {f: {m: fam_point(f, m, papers) for m in gains} for f, _, _ in FAMILIES}
rng, draws, fam_draws, fam_sft_draws = random.Random(0), defaultdict(list), defaultdict(list), defaultdict(list)
for _ in range(2000):
    sample = [rng.choice(papers) for _ in papers]
    for m in gains:
        for k, v in mq_point(m, sample).items():
            draws[(m, k)].append(v)
    for f, _, _ in FAMILIES:
        a, b = fam_point(f, "core-grpo-40", sample), fam_point(f, "sft-150", sample)
        if a is not None:
            fam_draws[f].append(a - b)
            fam_sft_draws[f].append(b - fam_point(f, "Qwen/Qwen3-8B", sample))
ci95 = lambda xs: [sorted(xs)[int(.025 * len(xs))], sorted(xs)[int(.975 * len(xs)) - 1]]  # noqa: E731
for (m, k), xs in draws.items():
    mq[m][k]["ci"] = ci95(xs)
fam_rl = {f: {"value": mq_family[f]["core-grpo-40"] - mq_family[f]["sft-150"], "ci": ci95(xs)} for f, xs in fam_draws.items()}
fam_sft = {f: {"value": mq_family[f]["sft-150"] - mq_family[f]["Qwen/Qwen3-8B"], "ci": ci95(xs)}
           for f, xs in fam_sft_draws.items()}


P = lambda m, k, scope="all": mq[m][k]["value"] if k in MQ else pooled[scope][m][k]["value"]  # noqa: E731
others = OPEN + OURS
models = [LUNA] + others

# Main table
cols = [("valid", "p", False, "all"), ("l3", "p", False, "all"), ("mq", "g", False, "all")]
cols += [(d, "g", False, "all") for d in DIMS]
cols += [("d1_balanced_accuracy", "p", False, "d1"), ("act_recall", "p", False, "d1"),
         ("over_edit_rate", "p", True, "d1"), ("pair_accuracy", "p", False, "d1")]


def value(m, k, scope):
    if m == "BioMistral/BioMistral-7B" and k == "over_edit_rate":
        return None  # it never states a decision
    return P(m, k, scope)


table = {}
for k, kind, lower, scope in cols:
    rank = [m for m in others if value(m, k, scope) is not None and not (k in MQ and m == OPEN[0])]  # BioMistral edits no map
    table[k] = dict(zip(models, cells(models, lambda m: value(m, k, scope), kind, lower, rank)))
    table[k][LUNA] = fmt(value(LUNA, k, scope), kind)
lines = []
for group in ([LUNA], OPEN, OURS):
    lines.append("\\midrule")
    for m in group:
        lines.append(f"{NAME[m]} & " + " & ".join(table[k][m] for k, *_ in cols) + " \\\\")
lines.append("\\midrule")
lines.append("\\textit{Reference} & -- & -- & " + " & ".join(f"\\textit{{{fmt(P(REF, k), 'g')}}}" for k in ["mq"] + DIMS)
             + " & -- & -- & -- & -- \\\\")
(TABLES / "main-results.tex").write_text(f"""% Pooled test results; {SOURCE} (pooled.json; generated by paper_tables.py).
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3.3pt}}
\\begin{{tabular}}{{@{{}}l ccc cccc cccc@{{}}}}
\\toprule
& \\multicolumn{{3}}{{c}}{{All 479 test tasks}} & \\multicolumn{{4}}{{c}}{{Map quality by rubric dimension}} & \\multicolumn{{4}}{{c}}{{108 decision cards}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-8}}\\cmidrule(l){{9-12}}
Model & Validity$\\uparrow$ & Correctness$\\uparrow$ & \\makecell{{Map\\\\quality$\\uparrow$}} & Clarity$\\uparrow$ & Coherence$\\uparrow$ & Traceab.$\\uparrow$ & Testab.$\\uparrow$ & BA$\\uparrow$ & Act$\\uparrow$ & Over$\\downarrow$ & Pair$\\uparrow$ \\\\
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\caption{\\textbf{Test results of all seven models.} Every criterion is computed on every task. Map quality: per task, the mean of the judge's eight comparisons of the updated map with the original ($+1$ better, 0 tie, $-1$ worse), averaged over tasks (dimensions: two comparisons each); an unchanged map, including one whose patch does not apply, scores 0. \\emph{Reference}: the reference answer's patch, judged the same way. BA: balanced accuracy; Act: act recall; Over: over-edit rate; Pair: minimal-pair accuracy. 95\\% intervals are in Table~\\ref{tab:ci}. GPT-6 Luna is the frontier model; SFT w/o pairs, SFT and SFT+RL are ours. \\textbf{Bold} and \\second{underlined}: best and second best among the other models.}
\\label{tab:main}
\\end{table*}
""")

# Confidence intervals (BioMistral omitted)
ci_models = [LUNA, "May2222/Fisher-R1-14B", "Qwen/Qwen3-8B"] + OURS
ci_rows = [("\\multicolumn{7}{@{}l}{\\textit{All 479 test tasks}}", None)]
ci_rows += [(label, (k, kind, "all")) for label, k, kind in
            [("Validity", "valid", "p"), ("Correctness", "l3", "p"), ("Map quality", "mq", "g"),
             ("\\quad Clarity", DIMS[0], "g"), ("\\quad Coherence", DIMS[1], "g"),
             ("\\quad Traceability", DIMS[2], "g"), ("\\quad Testability", DIMS[3], "g")]]
ci_rows += [("\\midrule\n\\multicolumn{7}{@{}l}{\\textit{108 decision cards}}", None)]
ci_rows += [(label, (k, "p", "d1")) for label, k in
            [("Balanced accuracy", "d1_balanced_accuracy"), ("Act recall", "act_recall"),
             ("Over-edit rate", "over_edit_rate"), ("Pair accuracy", "pair_accuracy")]]
out = []
for label, spec in ci_rows:
    if spec is None:
        out.append(label + " \\\\")
        continue
    k, kind, scope = spec
    parts = []
    for m in ci_models:
        x = mq[m][k] if k in MQ else pooled[scope][m][k]
        parts.append(f"{fmt(x['value'], kind)} [{fmt(x['ci'][0], kind)},{fmt(x['ci'][1], kind)}]")
    out.append(f"{label} & " + " & ".join(parts) + " \\\\")
(TABLES / "ci.tex").write_text(f"""% 95% paper-level bootstrap intervals (2,000 resamples of the 122 test papers) for Table~\\ref{{tab:main}};
% {SOURCE}/pooled.json (generated by paper_tables.py).
\\begin{{table*}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l c cc ccc@{{}}}}
\\toprule
& \\multicolumn{{1}}{{c}}{{Frontier}} & \\multicolumn{{2}}{{c}}{{Open models}} & \\multicolumn{{3}}{{c}}{{Ours}} \\\\
\\cmidrule(lr){{2-2}}\\cmidrule(lr){{3-4}}\\cmidrule(l){{5-7}}
& \\multicolumn{{1}}{{c}}{{GPT-6 Luna}} & \\multicolumn{{1}}{{c}}{{Fisher-R1-14B}} & \\multicolumn{{1}}{{c}}{{Qwen3-8B}} & \\multicolumn{{1}}{{c}}{{\\makecell{{SFT w/o\\\\pairs}}}} & \\multicolumn{{1}}{{c}}{{SFT}} & \\multicolumn{{1}}{{c}}{{SFT+RL}} \\\\
\\midrule
""" + "\n".join(out) + """
\\bottomrule
\\end{tabular*}
\\caption{\\textbf{95\\% confidence intervals for Table~\\ref{tab:main},} from 2,000 bootstrap resamples of the 122 test papers. BioMistral-7B, which returns no valid answer, is omitted.}
\\label{tab:ci}
\\end{table*}
""")

# Paired differences quoted in Section 5: pooled.json (validity, correctness, decision), the side-by-side judge's
# paired map-quality differences (data/results/map-quality/scores-gemini.json), and the RL development reward
# (step 40 minus SFT-150, Luna and Qwen3-8B mean, two samples per task, paired by task, 2,000 resamples of papers, seed 0)
mq_pairs = json.loads((ROOT / "data/results/map-quality/scores-gemini.json").read_text())["models"]


def pair(a, b, k, scope="all"):
    """a minus b with its 95% interval, from whichever of the two stores the paired difference."""
    store = (lambda m: mq_pairs[m]["gain"]) if k == "mq" else (lambda m: pooled[scope][m][k])
    if f"minus_{b}" in store(a):
        return store(a)[f"minus_{b}"]
    d = store(b)[f"minus_{a}"]
    return {"value": -d["value"], "ci": [-d["ci"][1], -d["ci"][0]]}


def d3(v):
    s = f"{abs(v):.3f}".lstrip("0")
    return f"${'+' if round(v, 3) > 0 else '-' if round(v, 3) < 0 else ''}{s}$"


dev = ROOT / "data/results/rl-dev/core-grpo40"
dev_paper, dev_score = {}, defaultdict(list)
for m in ("sft-150", "core-grpo-40"):
    for s in (1, 2):
        for line in gzip.open(ROOT / "data/tasks/development.jsonl.gz", "rt", encoding="utf-8"):
            meta = json.loads(line)["split_meta"]
            dev_paper[meta["task_id"]] = meta["paper_id"]
        luna = {r["task_id"]: r["reward"] for r in read(dev / f"{m}-r{s}.core-luna.jsonl")}
        qwen = {r["task_id"]: r["reward"] for r in read(dev / f"{m}-r{s}.core-qwen.jsonl")}
        for c in luna:
            dev_score[(m, c)].append((luna[c] + qwen[c]) / 2)
dev_diff = {c: st.mean(dev_score[("core-grpo-40", c)]) - st.mean(dev_score[("sft-150", c)])
            for m, c in dev_score if m == "sft-150"}
dev_by_paper = defaultdict(list)
for c in dev_diff:
    dev_by_paper[dev_paper[c]].append(c)
dev_papers, rng_dev, dev_draws = sorted(dev_by_paper), random.Random(0), []
for _ in range(2000):
    sample = [c for p in (rng_dev.choice(dev_papers) for _ in dev_papers) for c in dev_by_paper[p]]
    dev_draws.append(st.mean(dev_diff[c] for c in sample))
dev_pair = {"value": st.mean(dev_diff.values()), "ci": ci95(dev_draws)}

pair_rows = [("SFT $-$ Qwen3-8B", "sft-150", "Qwen/Qwen3-8B"), ("SFT $-$ Fisher-R1-14B", "sft-150", "May2222/Fisher-R1-14B"),
             ("SFT $-$ SFT w/o pairs", "sft-150", "sft-v1"), ("SFT+RL $-$ SFT", "core-grpo-40", "sft-150"),
             ("GPT-6 Luna $-$ SFT", LUNA, "sft-150")]
pair_cols = [("valid", "all"), ("l3", "all"), ("mq", "all"), ("d1_balanced_accuracy", "d1"), ("over_edit_rate", "d1")]
out = []
for label, a, b in pair_rows:
    out.append(f"{label} & " + " & ".join(
        f"{d3(x['value'])} [{d3(x['ci'][0])}, {d3(x['ci'][1])}]" for x in (pair(a, b, k, sc) for k, sc in pair_cols)) + " \\\\")
out.append("\\midrule")
out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{Development reward (Luna and Qwen3-8B mean; {len(dev_diff)} tasks, two samples each; "
           f"{len(dev_papers)} papers resampled), SFT+RL $-$ SFT: {d3(dev_pair['value'])} "
           f"[{d3(dev_pair['ci'][0])}, {d3(dev_pair['ci'][1])}]}} \\\\")
(TABLES / "paired-diffs.tex").write_text(f"""% Paired differences quoted in Section 5; {SOURCE}/pooled.json, data/results/map-quality/scores-gemini.json and
% data/results/rl-dev/core-grpo40 (generated by paper_tables.py).
\\begin{{table*}}[t]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l ccc cc@{{}}}}
\\toprule
& \\multicolumn{{3}}{{c}}{{All 479 test tasks}} & \\multicolumn{{2}}{{c}}{{108 decision cards}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(l){{5-6}}
Difference & Validity & Correctness & Map quality & Balanced accuracy & Over-edit rate \\\\
\\midrule
""" + "\n".join(out) + """
\\bottomrule
\\end{tabular*}
\\caption{\\textbf{Paired differences between models} on the same test tasks, with 95\\% intervals from 2,000 bootstrap resamples of the 122 test papers.}
\\label{tab:paired}
\\end{table*}
""")

# Per-family table
fam_models =[LUNA, "Qwen/Qwen3-8B", "May2222/Fisher-R1-14B"] + OURS
fam_rank = fam_models[1:]
F = lambda f, m, k: summary["families"][f][m][k]  # noqa: E731
out = []
for k, kind, title in (("valid", "p", "Validity"), ("l3", "p", "Correctness"), ("mq", "g", "Map quality ($-1$ to 1)")):
    out.append(("\\midrule\n" if out else "") + f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    for f, label, _ in FAMILIES:
        n = sum(1 for r in by_model["sft-150"] if r["family"] == f)
        fv = (lambda m: mq_family[f][m]) if k == "mq" else (lambda m: F(f, m, k)["value"])
        c = cells(fam_models, fv, kind, rank=fam_rank)
        c[0] = fmt(fv(LUNA), kind)
        q = F(f, "Qwen/Qwen3-8B", k)["minus_sft-150"] if k != "mq" else None  # Qwen minus SFT, so negate
        deltas = []
        for d in (fam_sft[f] if k == "mq" else {"value": -q["value"], "ci": q["ci"] and [-q["ci"][1], -q["ci"][0]]},
                  fam_rl[f] if k == "mq" else F(f, "core-grpo-40", k)["minus_sft-150"]):
            sig = "\\sig" if d["ci"] and (d["ci"][0] > 0 or d["ci"][1] < 0) else ""
            dv = round(d["value"], 2)
            deltas.append(("$+$" if dv > 0 else "$-$" if dv < 0 else "") + fmt(abs(dv), kind) + sig)
        out.append(f"{label} & {n} & " + " & ".join(c + deltas) + " \\\\")
(TABLES / "family-results.tex").write_text(f"""% Per-family test results; {SOURCE}/summary.json (generated by paper_tables.py).
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lr c cc ccc cc@{{}}}}
\\toprule
& & Frontier & \\multicolumn{{2}}{{c}}{{Open models}} & \\multicolumn{{3}}{{c}}{{Ours}} & & \\\\
\\cmidrule(lr){{3-3}}\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-8}}
Family & $n$ & GPT-6 Luna & Qwen3-8B & Fisher-R1-14B & SFT w/o pairs & SFT & SFT+RL & $\\Delta$ SFT & $\\Delta$ RL \\\\
\\midrule
""" + "\n".join(out) + """
\\bottomrule
\\end{tabular}
\\caption{\\textbf{Test results per family.} $n$: test tasks; $\\Delta$ SFT: SFT minus Qwen3-8B; $\\Delta$ RL: SFT+RL minus SFT; $^{*}$: the paired 95\\% interval excludes zero; $^\\dagger$: held out from RL. \\textbf{Bold} and \\second{underlined}: best and second best among the other models.}
\\label{tab:family-results}
\\end{table*}
""")


# Decision per family
def ba(rs):
    act = [r["d1"] for r in rs if r["label"] == "act"]
    no = [r["d1"] for r in rs if r["label"] == "no-act"]
    return st.mean([st.mean(x) for x in (act, no) if x])


d1_families = [("evidence_audit", "Evidence audit"), ("local_repair", "Local repair$^\\dagger$"),
               ("required_aspect_completion", "Aspect completion"), ("guided_preservation", "Guided change$^\\ddagger$")]
out = []
for f, label in d1_families:
    rs = {m: [r for r in by_model[m] if r["family"] == f and r["label"]] for m in fam_models}
    c = cells(fam_models, lambda m: ba(rs[m]), rank=fam_rank)
    c[0] = fmt(ba(rs[LUNA]))
    out.append(f"{label} & {len(rs[LUNA])} & " + " & ".join(c) + " \\\\")
assert {r["family"] for r in rows if r["label"]} == {f for f, _ in d1_families}
(TABLES / "decision-family.tex").write_text(f"""% Decision balanced accuracy per family; {SOURCE}/rows.jsonl (generated by paper_tables.py).
\\begin{{table}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{1.8pt}}
\\begin{{tabular}}{{@{{}}lr c cc ccc@{{}}}}
\\toprule
& & Front. & \\multicolumn{{2}}{{c}}{{Open}} & \\multicolumn{{3}}{{c}}{{Ours}} \\\\
\\cmidrule(lr){{3-3}}\\cmidrule(lr){{4-5}}\\cmidrule(l){{6-8}}
Family & $n$ & Luna & Qwen & Fisher & \\makecell{{w/o\\\\pairs}} & SFT & \\makecell{{SFT\\\\+RL}} \\\\
\\midrule
""" + "\n".join(out) + """
\\bottomrule
\\end{tabular}
\\caption{\\textbf{Decision balanced accuracy per family} on the test decision cards. $n$: cards; $^\\dagger$: held out from RL; $^\\ddagger$: all six are no-change cards, so the value is no-change recall. \\textbf{Bold} and \\second{underlined}: best and second best among the other models.}
\\label{tab:decision-family}
\\end{table}
""")

# Validity components and per-judge scores
extra = [("complete", "p", False), ("parse", "p", False), ("schema_valid", "p", False), ("patch_applies", "p", False),
         ("valid", "p", False), ("strict_valid", "p", False), ("coverage", "p", False), ("pitfall_rate", "p", True),
         ("improved", "p", False), ("worsened", "p", True)]


def ev(m, k):
    if k == "strict_valid":
        return st.mean(float(r["strict_valid"]) for r in by_model[m])
    return P(m, k)


# BioMistral asserts nothing and edits no map, so it is not ranked on forbidden inferences or map changes
cols_extra = {k: dict(zip(models, cells(models, lambda m: ev(m, k), kind, lower,
                                        [m for m in others if k not in ("pitfall_rate", "improved", "worsened") or m != OPEN[0]])))
              for k, kind, lower in extra}
for k, kind, _ in extra:
    cols_extra[k][LUNA] = fmt(ev(LUNA, k), kind)
out = []
for group in ([LUNA], OPEN, OURS):
    out.append("\\midrule")
    for m in group:
        out.append(f"{NAME[m]} & " + " & ".join(cols_extra[k][m] for k, *_ in extra) + " \\\\")
(TABLES / "pooled-extra.tex").write_text(f"""% Validity components, correctness items, map changes; {SOURCE} (generated by paper_tables.py).
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}l cccccc cc cc@{{}}}}
\\toprule
& \\multicolumn{{6}}{{c}}{{Validity components$\\uparrow$}} & \\multicolumn{{2}}{{c}}{{Correctness}} & \\multicolumn{{2}}{{c}}{{Map quality}} \\\\
\\cmidrule(lr){{2-7}}\\cmidrule(lr){{8-9}}\\cmidrule(l){{10-11}}
Model & Complete & Parses & Contract & Patch & Valid & Strict & Req.\\ met$\\uparrow$ & Forbid.$\\downarrow$ & Better$\\uparrow$ & Worse$\\downarrow$ \\\\
""" + "\n".join(out) + """
\\bottomrule
\\end{tabular}
\\caption{\\textbf{Validity components, correctness items and map changes} on all 479 test tasks. \\emph{Patch}: the patch applies; \\emph{Strict}: validity without forgiving meaning-preserving slips. \\emph{Req.\\ met}: share of requirements met (0 for a task with no answer text); \\emph{Forbid.}: share of forbidden inferences asserted, over answers with text. \\emph{Better}, \\emph{Worse}: share of tasks whose map-quality score, the mean of its eight comparisons, is above or below 0. Map quality in Table~\\ref{tab:main} averages the score itself, so a task scored, say, $+.25$ counts as 1 here but .25 there, and Better minus Worse need not equal map quality (GPT-6 Luna: $.36 - .25 = .11$, map quality .07). Rows as in Table~\\ref{tab:main}. \\textbf{Bold} and \\second{underlined}: best and second best among the other models.}
\\label{tab:pooled-extra}
\\end{table*}
""")

# Heatmap figure: correctness and map quality by family
RAMP = [(244, 249, 254), (237, 245, 252), (231, 241, 250), (225, 236, 248), (218, 232, 246), (212, 228, 244),
        (206, 224, 241), (199, 220, 239), (189, 214, 236), (177, 208, 233), (165, 203, 230), (153, 197, 227),
        (141, 191, 223), (129, 185, 220), (118, 179, 217), (106, 173, 214), (99, 166, 210), (91, 158, 207),
        (84, 150, 203), (77, 142, 200), (69, 135, 196), (62, 127, 193)]


def color(t):
    t = min(max(t, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    a, b = RAMP[i], RAMP[i + 1]
    c = [round(a[j] + (b[j] - a[j]) * (t - i)) for j in range(3)]
    return f"{{rgb,255:red,{c[0]};green,{c[1]};blue,{c[2]}}}"


fig_models = [LUNA] + OPEN + OURS
offsets = {LUNA: 0.0, "BioMistral/BioMistral-7B": 0.54, "May2222/Fisher-R1-14B": 0.96, "Qwen/Qwen3-8B": 1.38,
           "sft-v1": 1.92, "sft-150": 2.34, "core-grpo-40": 2.76}
xs = [1.22 * i for i in range(len(FAMILIES))] + [11.13]
lines = [f"% Per-family correctness and map quality on test; generated from {SOURCE} by paper_tables.py.",
         "% Light single-hue ramp per block; the outline marks SFT+RL.",
         "\\documentclass[11pt]{standalone}", "\\input{preamble}", "\\begin{document}",
         "\\begin{tikzpicture}[paper figure]"]
for x, (f, _, short) in zip(xs, FAMILIES + [("all", None, "\\textbf{All}")]):
    n = sum(1 for r in by_model["sft-150"] if f == "all" or r["family"] == f)
    lines.append(f"\\node[font=\\sffamily\\fontsize{{8}}{{9.5}}\\selectfont, align=center, anchor=south, inner sep=1pt] "
                 f"at ({x + 0.595:.3f},0.08) {{{short}\\\\{{\\color{{black!55}}n={n}}}}};")
blocks = [("l3", "Correctness (share of items passed)", 0.30, 1.0, "p", -0.22),
          ("mq", "Map quality (change over the original map)", -0.5, 1.0, "g", -4.10)]
for k, title, lo, hi, kind, head in blocks:
    lines.append(f"\\node[anchor=west, font=\\sffamily\\fontsize{{8.5}}{{10}}\\selectfont\\bfseries, inner sep=0pt] "
                 f"at (-2.45,{head:.3f}) {{{title}}};")
    for i in range(len(RAMP)):
        lines.append(f"\\fill[fill={color(i / (len(RAMP) - 1))}] ({10.12 + 0.1 * i:.3f},{head + 0.07:.3f}) rectangle ++(0.105,-0.15);")
    lines.append(f"\\node[anchor=east, font=\\sffamily\\scriptsize, text=black!60, inner sep=1.5pt] at (10.120,{head - 0.005:.3f}) {{{fmt(lo, kind) if kind == 'p' else '$-$0.5' if kind == 'g' else f'{lo:.1f}'}}};")
    lines.append(f"\\node[anchor=west, font=\\sffamily\\scriptsize, text=black!60, inner sep=1.5pt] at (12.320,{head - 0.005:.3f}) {{{'1.0' if kind == 'p' else f'{hi:.1f}'}}};")
    top0 = head - 0.28
    # map quality also shows the reference answers' own gain, on top (correctness of a reference is ~1 by construction)
    rows = {REF: 0.0, **{m: o + 0.54 for m, o in offsets.items()}} if k == "mq" else offsets
    for m in rows:
        top = top0 - rows[m]
        style = "\\bfseries" if m in ("sft-150", "core-grpo-40") else "\\itshape" if m == REF else ""
        label = "Reference" if m == REF else NAME[m]
        lines.append(f"\\node[anchor=east, font=\\sffamily\\fontsize{{8.5}}{{10}}\\selectfont{style}] at (-0.1,{top - 0.21:.3f}) {{{label}}};")
        for x, (f, _, _) in zip(xs, FAMILIES + [("all", None, None)]):
            v = P(m, k) if f == "all" else mq_family[f][m] if k == "mq" else F(f, m, k)["value"]
            lines.append(f"\\fill[fill={color((v - lo) / (hi - lo))}] ({x:.3f},{top:.3f}) rectangle ++(1.190,-0.390);")
            lines.append(f"\\node[font=\\sffamily\\scriptsize, text=black!85] at ({x + 0.595:.3f},{top - 0.195:.3f}) {{{fmt(v, kind)}}};")
    top = top0 - rows["core-grpo-40"]
    lines.append(f"\\draw[black!75, line width=0.7pt, rounded corners=1pt] (-0.03,{top + 0.03:.3f}) rectangle (12.350,{top - 0.42:.3f});")
lines += ["\\end{tikzpicture}", "\\end{document}"]
FIGURE.write_text("\n".join(lines) + "\n")
print("wrote", ", ".join(p.name for p in sorted(TABLES.glob("*.tex"))), "and", FIGURE.name)
