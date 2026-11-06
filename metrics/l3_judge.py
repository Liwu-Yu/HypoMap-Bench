#!/usr/bin/env python3
"""L3 reference correctness: a closed check of an answer against its card's reviewed items
(docs/specs/metrics.md §6).

One call per answer. The judge sees the public packet, the final answer, the card's answer
requirements and forbidden inferences, and its acceptable alternatives (never the reference
answer), and says for each item whether the answer meets the requirement or asserts the
inference. Card score: mean over items, a met requirement and an absent pitfall each 1.

    python metrics/l3_judge.py --answers A.jsonl --output scores.jsonl [--model gpt-6-luna] [--samples 1]
    python metrics/l3_judge.py --answers A.jsonl --export-jobs JOBS.jsonl [--samples 2]   # for a judge pod
    python metrics/l3_judge.py --jobs JOBS.jsonl --resume --output scores.jsonl --list-ids --export-jobs RETRY.jsonl
                                                     # the calls still without a verdict, with the item IDs listed
    python metrics/l3_judge.py --jobs JOBS.jsonl --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B \
        --judge qwen3-8b --max-tokens 16384 --output scores.jsonl --resume

answers JSONL: {"id", "task_id", "answer"}; answer is the parsed final-answer JSON. The card's
packet and items are found by task_id (split v2 exports, decision-variants ledger, snapshot).
A jobs file holds each call's prepared message and item IDs, so a judge pod needs no repository data.
With --base-url the judge is a vLLM server (Qwen3-8B) with the L4 settings (thinking, temperature 0.6,
top-p 0.95). `judge()` is importable for the GRPO reward.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data/tasks"

SYSTEM = (
    "You check a model's answer to a scientific workspace task against the task's reviewed answer items. "
    "This is a closed check: judge only the listed items, and do not question the items themselves.\n\n"
    "You get the public task packet, the model's final answer (JSON), the items, and acceptable alternatives: "
    "other ways of answering that reviewers accepted as valid.\n"
    "- Requirement items (R1, R2, ...): \"yes\" if the answer meets the requirement, directly or through an "
    "acceptable alternative; \"no\" if it does not, misses its main point, or contradicts it. Judge substance, "
    "not wording or layout: the answer need not restate every value, ID or example the requirement lists, "
    "unless the requirement is about that detail. A requirement to change the map (an update, a link, a "
    "refinement) is met only if the answer's proposed updates actually make that change.\n"
    "- Pitfall items (P1, P2, ...): \"yes\" if the answer asserts the forbidden inference, or something that "
    "implies it, anywhere in the answer. Raising it in order to reject it, or stating it as an untested "
    "possibility, is not asserting it.\n\n"
    "Read the answer under the packet's response contract. Aspect nodes are propositions to be assessed: an "
    "added or revised aspect statement proposes what to test and does not assert that it holds. The answer "
    "asserts a result only where it presents it as established, as in assessments, findings, rationales and "
    "reasons. No operation deletes map nodes: one aspect is consolidated into another, or retired, by revising "
    "the kept node and detaching the other.\n\n"
    'Return only a JSON object: {"items": [{"id": "R1", "verdict": "yes" or "no", "evidence": "a short exact '
    'quote from the answer, or empty", "reason": "one sentence"}]}, with every item exactly once.'
)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cards():
    """task_id -> {packet, reference, bundle (the reviewed items), task_family, split, d1} for every task."""
    cards = {}
    for split in ("development", "test", "sft", "rl"):
        for row in read_jsonl(EXPORTS / f"{split}.jsonl.gz"):
            meta = row["split_meta"]
            cards[meta["task_id"]] = {"packet": row["input"], "reference": row["output"], "bundle": row["items"],
                                      "task_family": meta["task_family"], "split": split, "d1": meta["d1"]}
    return cards


def items(bundle):
    out = [{"id": f"R{i}", "type": "requirement", "requirement": r.get("requirement"), "basis_ids": r.get("basis_ids")}
           for i, r in enumerate(bundle.get("answer_requirements") or [], 1)]
    out += [{"id": f"P{i}", "type": "pitfall", "forbidden_inference": p.get("inference"), "why_forbidden": p.get("reason")}
            for i, p in enumerate(bundle.get("forbidden_inferences") or [], 1)]
    return out


def message(packet, answer, bundle):
    return json.dumps({"public_packet": packet, "model_answer": answer, "items": items(bundle),
                       "acceptable_alternatives": bundle.get("acceptable_alternatives") or []}, ensure_ascii=False)


def id_reminder(ids):
    """Appended to a call's message with --list-ids: Qwen3-8B often leaves the pitfall items out of its JSON."""
    return "\n\nReturn a verdict for every listed item, in this order: " + ", ".join(ids) + "."


def parse(raw, ids):
    """{item id: True if met (requirement) / asserted (pitfall)}, or None unless every item has a verdict."""
    text = raw.rsplit("</think>", 1)[-1]
    start = text.find("{")
    try:
        obj = json.JSONDecoder().raw_decode(text[start:])[0] if start >= 0 else None
        verdicts = {row["id"]: row["verdict"].strip().lower() for row in obj["items"]}
    except (ValueError, KeyError, TypeError, AttributeError):
        # Quotes copied into "evidence" sometimes break the JSON; the verdicts precede them.
        verdicts = dict(re.findall(r'"id"\s*:\s*"([RP]\d+)"\s*,\s*"verdict"\s*:\s*"(yes|no)"', text))
    if any(verdicts.get(i) not in ("yes", "no") for i in ids):
        return None
    return {i: verdicts[i] == "yes" for i in ids}


def score(verdicts):
    reqs = [v for i, v in verdicts.items() if i.startswith("R")]
    pits = [v for i, v in verdicts.items() if i.startswith("P")]
    good = reqs + [not v for v in pits]
    return {"l3": sum(good) / len(good) if good else None,
            "coverage": sum(reqs) / len(reqs) if reqs else None,
            "pitfall_rate": sum(pits) / len(pits) if pits else None}


def judge(client, model, packet, answer, bundle, max_tokens=8192, extra=None):
    """One closed-check call; returns verdicts (or None), scores and the raw reply."""
    return call(client, model, message(packet, answer, bundle), [i["id"] for i in items(bundle)], max_tokens,
                extra=extra)


def openrouter(provider, reasoning=False):
    """Client and request body for an OpenRouter judge pinned to one provider, without data collection."""
    import os
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=900)
    extra = {"provider": {"order": [provider], "allow_fallbacks": False, "data_collection": "deny"},
             "usage": {"include": True}}
    if reasoning:
        extra["reasoning"] = {"enabled": True}
    return client, extra


def call(client, model, user, ids, max_tokens=8192, chat=False, extra=None):
    """One call on a prepared message. chat: a vLLM chat server (Qwen3-8B with the L4 judge settings);
    extra: an OpenRouter request body (pinned provider, default sampling and reasoning)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if extra is not None:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                  extra_body=extra)
        raw = response.choices[0].message.content or ""
    elif chat:
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.6, top_p=0.95,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        raw = response.choices[0].message.content or ""
    else:
        response = client.responses.create(model=model, store=False, max_output_tokens=max_tokens, input=messages)
        raw = response.output_text or ""
    verdicts = parse(raw, ids)
    return {"verdicts": verdicts, **(score(verdicts) if verdicts else {}), "raw": raw,
            "usage": response.usage.model_dump() if response.usage else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--answers", type=Path)
    source.add_argument("--jobs", type=Path, help="Prepared calls from --export-jobs.")
    parser.add_argument("--export-jobs", type=Path, help="Write the calls and exit (with --resume, only unfinished ones).")
    parser.add_argument("--list-ids", action="store_true", help="End each message with the item IDs to judge.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--judge", help="Label written to each row (default: --model).")
    parser.add_argument("--base-url", help="OpenAI-compatible chat server (vLLM); omit for the OpenAI Responses API.")
    parser.add_argument("--provider", help="OpenRouter provider tag to pin (e.g. google-ai-studio/flex); the "
                                           "model is then an OpenRouter model ID, with no fallback and no data collection.")
    parser.add_argument("--reasoning", action="store_true",
                        help="OpenRouter: turn thinking on explicitly (some DeepSeek hosts default to off).")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Append to --output, skipping finished calls.")
    args = parser.parse_args()

    if args.answers:
        cards = load_cards()
        jobs = [{"id": row["id"], "task_id": row["task_id"], "sample": s,
                 "message": message(cards[row["task_id"]]["packet"], row["answer"], cards[row["task_id"]]["bundle"]),
                 "item_ids": [i["id"] for i in items(cards[row["task_id"]]["bundle"])]}
                for s in range(args.samples) for row in read_jsonl(args.answers) if isinstance(row.get("answer"), dict)]
    else:
        jobs = read_jsonl(args.jobs)
    if args.list_ids:
        jobs = [{**job, "message": job["message"] + id_reminder(job["item_ids"])} for job in jobs]
    done = set()
    if args.resume and args.output.exists():  # failed or unparsed calls are retried
        done = {(r["id"], r["sample"]) for r in read_jsonl(args.output) if r.get("verdicts")}
    jobs = [job for job in jobs if (job["id"], job["sample"]) not in done]
    if args.export_jobs:
        with gzip.open(args.export_jobs, "wt", encoding="utf-8") if str(args.export_jobs).endswith(".gz") \
                else args.export_jobs.open("w", encoding="utf-8") as sink:
            for job in jobs:
                sink.write(json.dumps(job, ensure_ascii=False) + "\n")
        print(f"{args.export_jobs}: {len(jobs)} calls")
        return
    print(f"{len(jobs)} calls", flush=True)

    from openai import OpenAI
    if not args.base_url:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    extra = None
    if args.provider:
        client, extra = openrouter(args.provider, args.reasoning)
    elif args.base_url:
        client = OpenAI(base_url=args.base_url, api_key="local", timeout=7200)
    else:
        client = OpenAI()

    def run(job):
        try:
            result = call(client, args.model, job["message"], job["item_ids"], args.max_tokens,
                          chat=bool(args.base_url), extra=extra)
        except Exception as error:  # left out of the output; a --resume run retries it
            print(job["id"], job["sample"], "error", repr(error)[:200], flush=True)
            return None
        return {"id": job["id"], "task_id": job["task_id"], "sample": job["sample"],
                "judge": args.judge or args.model, **result}

    with args.output.open("a" if args.resume else "x", encoding="utf-8") as sink, \
            ThreadPoolExecutor(args.concurrency) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            row = future.result()
            if row is None:
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(row["id"], row["sample"], "ok" if row["verdicts"] else "unparsed", flush=True)


if __name__ == "__main__":
    main()
