#!/usr/bin/env python3
"""Render a HypoMap-Bench workspace as a compact document for the L4 judge (docs/specs/metrics.md §7, §11 step 2),
in the layout of metrics/render_session.py (main hypothesis, sub-hypotheses, experiments).

The workspace is a task packet's map, observations and experiment plans; render the public packet as it is
(pre-patch) or after `apply_proposed_updates` (a model's or the reference's patch). Deterministic, no model
summarization. Kept: the objective (main hypothesis or question, scope, validation requirements), every map
node with its kind and its map links, the experiment plans with their design and what they target, and the
observations with their scope and comparison. Dropped: the task question, response contract, user guidance,
background packet, constraints and existing interpretations (the task, not the workspace), and provenance
(source spans, evidence-basis tags, inference reasons), which a patch does not carry, so added and original
nodes read alike. IDs become labels: nodes H1, H2, ... in hierarchy order (breadth-first from the objective
over composition and refinement links), experiments E1, ..., observations O1, ...

    python metrics/render_workspace.py --split test --task-id TASK_ID   # print one pre-patch workspace
"""

import argparse
import re

HIERARCHY = ("composition", "refinement")


def text(value):
    return " ".join(str(value).split()) if value else ""


def comparison(c):
    if not isinstance(c, dict):
        return ""
    parts = []
    for key, name in [("groups", "groups"), ("endpoint", "endpoint"), ("factors_varied", "varied"),
                      ("factors_held_fixed", "held fixed"), ("analysis_unit", "unit"), ("measurement_window", "window"),
                      ("units", "units"), ("result", "result"), ("uncertainty", "uncertainty")]:
        value = c.get(key)
        if isinstance(value, list):
            value = "; ".join(text(v) for v in value if v)
        if value:
            parts.append(f"{name}: {text(value)}")
    return " | ".join(parts)


def merged(packet, updates):
    """Nodes merged away by a consolidation: targets of PROPOSE_CONSOLIDATION left with no link in the patched
    packet. No operation deletes a node, so a consolidation detaches the duplicate, which would otherwise be
    rendered as an unlinked aspect."""
    linked = {i for e in packet["current_map"]["edges"] for i in (e["source_id"], e["target_id"])}
    nodes = {n["id"] for n in packet["current_map"]["nodes"]}
    return {i for u in updates or [] if isinstance(u, dict) and u.get("operation") == "PROPOSE_CONSOLIDATION"
            for i in u.get("target_ids") or [] if i in nodes and i not in linked}


def render(packet, hide=()):
    """hide: node IDs left out (merged(); used by the side-by-side judge, data/results/map-quality)."""
    objective, graph = packet["objective"], packet["current_map"]
    nodes, edges = [n for n in graph["nodes"] if n["id"] not in hide], graph["edges"]
    plans, observations = packet.get("experiment_plans") or [], packet.get("visible_observations") or []

    # Node order: breadth-first from the objective over hierarchy links (either direction), then the rest.
    neighbours = {}
    for e in edges:
        if e["type"] in HIERARCHY:
            neighbours.setdefault(e["source_id"], []).append(e["target_id"])
            neighbours.setdefault(e["target_id"], []).append(e["source_id"])
    node_ids = [n["id"] for n in nodes]
    order, queue, seen = [], [objective["id"]], {objective["id"]}
    while queue:
        for nxt in neighbours.get(queue.pop(0), []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
                if nxt in node_ids:
                    order.append(nxt)
    order += [i for i in node_ids if i not in seen]
    labels = {objective["id"]: "Main hypothesis"}
    labels.update({i: f"H{k}" for k, i in enumerate(order, 1)})
    labels.update({p["id"]: f"E{k}" for k, p in enumerate(plans, 1)})
    labels.update({o["id"]: f"O{k}" for k, o in enumerate(observations, 1)})
    for k, r in enumerate(objective.get("validation_requirements") or [], 1):
        labels.setdefault(r["id"], f"R{k}")

    def label(i):
        return labels.get(i, "[unlisted]")

    links = {}
    for e in edges:
        kind = e["type"].replace("_", " ")
        links.setdefault(e["source_id"], []).append(f"{kind} → {label(e['target_id'])}")
        links.setdefault(e["target_id"], []).append(f"{kind} ← {label(e['source_id'])}")

    scope = objective.get("scientific_system_and_scope") or {}
    lines = ["# Main hypothesis", text(objective.get("main_hypothesis_or_question"))]
    if scope.get("system"):
        lines.append(f"System: {text(scope['system'])}")
    for key, name in [("conditions", "Conditions"), ("scope_limits", "Scope limits")]:
        if scope.get(key):
            lines.append(f"{name}: " + "; ".join(text(x) for x in scope[key]))
    if links.get(objective["id"]):
        lines.append("Links: " + "; ".join(links[objective["id"]]))
    requirements = objective.get("validation_requirements") or []
    if requirements:
        lines.append("Validation requirements:")
        lines += [f"- {label(r['id'])}: {text(r.get('statement'))}"
                  + (f" (aspects: {', '.join(label(a) for a in r['aspect_ids'])})" if r.get("aspect_ids") else "")
                  for r in requirements]
    lines += ["", "# Sub-hypotheses"]
    by_id = {n["id"]: n for n in nodes}
    for i in order:
        n = by_id[i]
        lines.append(f"## {label(i)} ({text(n.get('type'))})")
        lines.append(f"Statement: {text(n.get('statement'))}")
        if links.get(i):
            lines.append("Links: " + "; ".join(links[i]))
        lines.append("")

    lines.append("# Experiments")
    by_plan = {}
    for o in observations:
        by_plan.setdefault(o.get("experiment_id"), []).append(o)
    for p in plans:
        lines.append(f"## {label(p['id'])}")
        lines.append(f"Statement: {text(p.get('statement'))}")
        if p.get("manipulation"):
            lines.append(f"Manipulation: {text(p['manipulation'])}")
        if p.get("measurements"):
            lines.append("Measurements: " + "; ".join(text(x) for x in p["measurements"]))
        if comparison(p.get("comparison")):
            lines.append(f"Comparison: {comparison(p['comparison'])}")
        if links.get(p["id"]):
            lines.append("Links: " + "; ".join(links[p["id"]]))
        observed = [label(o["id"]) for o in by_plan.get(p["id"], [])]
        if observed:
            lines.append("Observations: " + ", ".join(observed))
        lines.append("")

    lines.append("# Observations")
    for o in observations:
        lines.append(f"## {label(o['id'])}" + (f" (from {label(o['experiment_id'])})" if o.get("experiment_id") in labels else ""))
        lines.append(f"Statement: {text(o.get('statement'))}")
        if o.get("scope"):
            lines.append(f"Scope: {text(o['scope'])}")
        if comparison(o.get("comparison")):
            lines.append(f"Comparison: {comparison(o['comparison'])}")
        if o.get("target_aspect_ids"):
            lines.append("Bears on: " + ", ".join(label(a) for a in o["target_aspect_ids"]))
        if links.get(o["id"]):
            lines.append("Links: " + "; ".join(links[o["id"]]))
        lines.append("")

    doc = "\n".join(lines).strip() + "\n"
    # IDs quoted inside statements become the document's labels.
    pattern = re.compile(r"(?<![\w-])(" + "|".join(re.escape(i) for i in sorted(labels, key=len, reverse=True)) + r")(?![\w-])")
    return pattern.sub(lambda m: labels[m.group(1)], doc)


def main():
    import gzip
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    with gzip.open(root / f"data/tasks/{args.split}.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split_meta"]["task_id"] == args.task_id:
                print(render(row["input"]))
                return
    raise SystemExit(f"{args.task_id} not in {args.split}")


if __name__ == "__main__":
    main()
