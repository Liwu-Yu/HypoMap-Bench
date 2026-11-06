#!/usr/bin/env python3
"""Absolute L4 workspace-quality judge (docs/specs/metrics.md §7). Not used in the paper: the paper's map quality is
metrics/map_quality.py, which takes the rubric (DIMENSIONS) and first_json from this file.

Scores each workspace on the four dimensions of the expert rubric (1-7) with expert-rated
few-shot examples. Scientific validity and feasibility are not judged: they need
domain and laboratory knowledge the judge models do not have. One judge per run; the panel is the mean of two runs.

    # GPT-6 Luna through the OpenAI API (OPENAI_API_KEY)
    python metrics/l4_judge.py --judge luna --model gpt-6-luna \\
        --example-chars 12000 --workspaces W.jsonl --examples E.jsonl --output scores-luna.jsonl
    # Qwen3-8B through a local vLLM server
    python metrics/l4_judge.py --judge qwen3-8b --model Qwen/Qwen3-8B \\
        --base-url http://127.0.0.1:8000/v1 --example-chars 12000 \\
        --workspaces W.jsonl --examples E.jsonl --output scores-qwen.jsonl

Inputs (JSONL):
  workspaces: {"workspace_id", "group", "text", optional "rotation"}
  examples:   {"workspace_id", "group", "text", "scores": {dim: 1-7}, "reasons": {dim: str}}
              (reasons from metrics/l4_example_reasons.py)
An example is never shown when scoring a workspace of the same group (paper or
participant), so the expert-rated workspaces can be scored leave-one-group-out.
A workspace's "rotation" offsets the example rotation: with one sample per workspace, test workspaces of
different cards then see different example sets, and the workspaces of one card (each model's) the same.
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
from pathlib import Path

DIMENSIONS = {
    "hypothesis_clarity": (
        "Hypothesis clarity",
        "Are the main hypothesis and sub-hypotheses stated clearly and precisely? High-quality plans explicitly "
        "state the main hypothesis, express each sub-hypothesis as a distinct scientific claim, specify relevant "
        "mechanisms, variables, populations, and conditions, and avoid vague or ambiguous language."),
    "hierarchical_coherence": (
        "Hierarchical coherence",
        "Does the plan organize the main hypothesis and sub-hypotheses into a clear and logically coherent "
        "hierarchy? High-quality plans decompose the main claim into meaningful and increasingly specific "
        "sub-hypotheses, maintain clear parent-child relationships, distinguish rather than repeat branches, and "
        "avoid contradictions or disconnected components."),
    "traceability": (
        "Evidence-hypothesis traceability",
        "Can each hypothesis and validation decision be traced to its supporting rationale, experiment, and "
        "evidence? High-quality plans explain why each element is included, connect literature and assumptions to "
        "the claims they support, identify which experiment or result bears on each sub-hypothesis, and show how "
        "the evidence informs the broader hypothesis."),
    "testability": (
        "Testability",
        "Can the proposed sub-hypotheses be evaluated using observable and distinguishable evidence? High-quality "
        "plans specify measurable variables or outcomes, derive observable predictions, define evidence that could "
        "support, weaken, or leave a claim unresolved, and distinguish competing sub-hypotheses where possible."),
}

# How to read the criteria for a plan written before any experiment is run (the experts
# rated validation plans written before any experiment). No dimension-specific notes: in the alignment runs they added nothing or lowered it.
INTRO = (
    "Each workspace is a hypothesis-validation plan written before any experiment is run, so its hypotheses are "
    "untested by design. Judge how well the plan states, organizes, links and designs the validation, not whether "
    "the hypotheses are true or already established by the cited literature."
)

SYSTEM = (
    "You are a blinded expert reviewer of scientific hypothesis-validation workspaces. Rate the workspace on each "
    "dimension from 1 (very poor) to 7 (excellent), using the questions and criteria below exactly as an expert "
    "judge would. Judge only the workspace shown; you are not told who or what produced it.\n\n"
    + "\n".join(f"- {key} ({name}): {criteria}" for key, (name, criteria) in DIMENSIONS.items())
    + "\n\n" + INTRO
    + "\n\nUse the whole 1-7 scale against the criteria: 7 = meets every criterion with no notable gap; "
      "6 = meets the criteria with minor gaps; 5 = meets most criteria with some clear gaps; 4 = mixed; "
      "3 = several criteria unmet; 2 = most criteria unmet; 1 = the dimension is essentially absent. "
      "Rate what the workspace contains, not its length."
    + "\n\nReturn only a JSON object: {\"scores\": {dimension: integer 1-7}, \"reasons\": {dimension: one sentence}} "
      f"with the keys {', '.join(DIMENSIONS)}."
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def user_message(workspace, examples, example_chars):
    parts = []
    for i, ex in enumerate(examples, 1):
        text = ex["text"] if not example_chars or len(ex["text"]) <= example_chars else ex["text"][:example_chars] + "\n[...]"
        rated = {"scores": {d: ex["scores"][d] for d in DIMENSIONS}}
        if ex.get("reasons"):
            rated["reasons"] = {d: r for d, r in ex["reasons"].items() if d in DIMENSIONS}
        parts.append(f"## Expert-rated example {i}\n{text}\n\nExpert rating:\n{json.dumps(rated, ensure_ascii=False)}")
    parts.append(f"## Workspace to rate\n{workspace['text']}")
    return "\n\n".join(parts)


def pick_examples(pool, k, sample=0):
    """Spread examples over the expert score range: sort by mean expert score, cut into k bins of neighbouring
    scores and take one example from each bin; successive samples rotate through each bin, so a workspace's
    score averages over several example sets rather than depending on one."""
    pool = sorted(pool, key=lambda ex: sum(ex["scores"][d] for d in DIMENSIONS) / len(DIMENSIONS))
    if len(pool) <= k:
        return pool
    bins = [pool[round(i * len(pool) / k):round((i + 1) * len(pool) / k)] for i in range(k)]
    return [b[sample % len(b)] for b in bins]


def first_json(raw):
    text = raw.rsplit("</think>", 1)[-1].strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]  # first JSON object; ignore trailing text
    except json.JSONDecodeError:
        return None


def parse(raw):
    obj = first_json(raw)
    try:
        scores = {k: int(obj["scores"][k]) for k in DIMENSIONS}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(1 <= v <= 7 for v in scores.values()):
        return None
    return {"scores": scores, "reasons": obj.get("reasons", {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", required=True, help="Label written to each row, e.g. luna or qwen3-8b.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="OpenAI-compatible server (vLLM); omit for the OpenAI Responses API.")
    parser.add_argument("--provider", help="OpenRouter provider tag to pin (e.g. google-ai-studio/flex); the model is "
                                           "then an OpenRouter model ID, with no fallback and no data collection.")
    parser.add_argument("--workspaces", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--example-chars", type=int, default=0, help="Truncate example text (0 = full).")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Append to --output, skipping finished calls.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt; no calls.")
    args = parser.parse_args()

    workspaces, examples = read_jsonl(args.workspaces), read_jsonl(args.examples)
    done = set()
    if args.resume and args.output.exists():
        done = {(r["workspace_id"], r["sample"]) for r in read_jsonl(args.output) if r.get("parsed")}  # retry the rest
    jobs = []
    for s in range(args.samples):  # sample-major: calls of one sample share the examples, so a server can reuse the prefix
        for ws in workspaces:
            if (ws["workspace_id"], s) in done:
                continue
            shots = pick_examples([ex for ex in examples if ex["group"] != ws["group"]], args.max_examples,
                                  ws.get("rotation", 0) + s)
            jobs.append((ws, [ex["workspace_id"] for ex in shots], user_message(ws, shots, args.example_chars), s))
    if args.dry_run:
        print(SYSTEM + "\n\n" + jobs[0][2])
        print(f"\n{len(jobs)} calls; first prompt {len(SYSTEM) + len(jobs[0][2])} characters")
        return

    from openai import OpenAI
    if not args.base_url:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if args.provider:
        import os
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=900)
    else:
        client = OpenAI(base_url=args.base_url, api_key="local", timeout=7200) if args.base_url else OpenAI()

    def call(job):
        try:
            return judge(job)
        except Exception as error:  # left out of the output; a --resume run retries it
            print(job[0]["workspace_id"], job[3], "error", repr(error)[:200], flush=True)
            return None

    def judge(job):
        ws, shot_ids, message, sample = job
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
        if args.provider:  # model defaults for sampling and reasoning, as for Luna
            response = client.chat.completions.create(
                model=args.model, messages=messages, max_tokens=args.max_tokens,
                extra_body={"provider": {"order": [args.provider], "allow_fallbacks": False, "data_collection": "deny"},
                            "usage": {"include": True}})
            raw = response.choices[0].message.content or ""
        elif args.base_url:
            response = client.chat.completions.create(
                model=args.model, messages=messages, max_tokens=args.max_tokens,
                temperature=args.temperature, top_p=0.95,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}})
            raw = response.choices[0].message.content or ""
        else:
            response = client.responses.create(model=args.model, input=messages, store=False,
                                               max_output_tokens=args.max_tokens)
            raw = response.output_text
        return {"workspace_id": ws["workspace_id"], "judge": args.judge, "model": args.model, "sample": sample,
                "examples": shot_ids, "parsed": parse(raw), "raw": raw,
                "usage": response.usage.model_dump() if response.usage else None}

    with args.output.open("a" if args.resume else "x", encoding="utf-8") as sink, \
            ThreadPoolExecutor(args.concurrency) as pool:
        for row in map(Future.result, as_completed([pool.submit(call, job) for job in jobs])):
            if row is None:
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(row["workspace_id"], row["sample"], "ok" if row["parsed"] else "unparsed", flush=True)


if __name__ == "__main__":
    main()
