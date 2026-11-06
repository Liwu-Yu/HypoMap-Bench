#!/usr/bin/env python3
"""Run the prepared HypoMap-Bench cards with OpenAI or local Hugging Face models.

    python inference.py --dry-run
    python inference.py --limit 1
    python inference.py --model qwen3-8b --dataset papers --limit 1
    python inference.py --model biomistral-7b --base-url http://localhost:8000/v1

API dependencies: openai, python-dotenv. Local dependencies: torch>=2.6,
transformers>=4.51.3,<5, accelerate. See README.md for setup.
No HypoMap-Bench package imports, reference answers, grading, or tool execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parent
MODELS = {
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-6-luna": "gpt-6-luna",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "biomistral-7b": "BioMistral/BioMistral-7B",
    "fisher-r1-14b": "May2222/Fisher-R1-14B",
}
OPENAI_MODELS = {"gpt-5.6-luna", "gpt-6-luna"}
LUNA_PRICING = {
    "checked_on": "2026-09-21",
    "source": "https://developers.openai.com/api/docs/pricing",
    "currency": "USD",
    "service_tier": "default",
    "usd_per_million_tokens": {
        "uncached_input": 0.20, "cached_input": 0.02,
        "cache_write_input": 0.25, "output": 1.20,
    },
    "long_context_above_input_tokens": 272000,
}
OPENAI_PRICING = {
    "gpt-5.6-luna": LUNA_PRICING,
    "gpt-6-luna": {
        **LUNA_PRICING,
        "checked_on": "2026-09-28",
        "usd_per_million_tokens": {
            "uncached_input": 0.10, "cached_input": 0.01,
            "cache_write_input": 0.125, "output": 0.50,
        },
    },
}


def estimate_cost(result: dict) -> dict | None:
    """Published-rate estimate from returned usage, not an invoiced charge."""
    usage = result.get("usage")
    pricing = OPENAI_PRICING.get(result.get("returned_model"))
    if (not usage or pricing is None
            or result.get("service_tier") != "default"):
        return None
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    written = details.get("cache_write_tokens", 0) or 0
    uncached = usage["input_tokens"] - cached - written
    if uncached < 0:
        return None
    rates = pricing["usd_per_million_tokens"]
    long_context = usage["input_tokens"] > pricing["long_context_above_input_tokens"]
    input_multiplier, output_multiplier = (2, 1.5) if long_context else (1, 1)
    components = {
        "uncached_input_usd": uncached * rates["uncached_input"] * input_multiplier / 1e6,
        "cached_input_usd": cached * rates["cached_input"] * input_multiplier / 1e6,
        "cache_write_input_usd": written * rates["cache_write_input"] * input_multiplier / 1e6,
        "output_usd": usage["output_tokens"] * rates["output"] * output_multiplier / 1e6,
    }
    return {
        "estimated_usd": round(sum(components.values()), 10),
        "components": {k: round(v, 10) for k, v in components.items()},
        "input_tokens_uncached": uncached, "input_tokens_cached": cached,
        "input_tokens_cache_write": written, "long_context": long_context,
        "cache_write_count_reported": "cache_write_tokens" in details,
        "note": "Published-rate estimate; unreported cache writes count as zero. Excludes taxes, credits, negotiated rates, and unreturned usage.",
        "pricing": pricing,
    }


def read_cards(data_dir: Path, dataset: str) -> list[dict]:
    names = ("papers",) if dataset == "all" else (dataset,)
    cards = []
    for name in names:
        with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                cards.append(json.loads(line))
    return cards


def read_completed(paths: list[Path], cards: list[dict], model_id: str, backend: str) -> dict[str, dict]:
    """Reuse completed answers by task ID; source attempt logs remain unchanged."""
    selected_ids = {card["card_id"] for card in cards}
    completed = {}
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                # A killed writer can leave its last JSONL record half-written.
                if line_number == len(lines) and not line.endswith("\n"):
                    print(f"Ignoring unfinished last record in {path}:{line_number}", file=sys.stderr)
                    continue
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(result, dict) or not result.get("card_id"):
                raise ValueError(f"Missing card_id in {path}:{line_number}")
            card_id = result["card_id"]
            if card_id not in selected_ids:
                continue
            if result.get("requested_model") != model_id or result.get("backend") != backend:
                raise ValueError(f"Model/backend mismatch in {path}:{line_number} for {card_id}")
            if result.get("status") == "completed":
                # For overlapping files, the last completed record supplied wins.
                completed[card_id] = {**result, "resumed_from": str(path.resolve())}
    return completed


def model_messages(card: dict, model: str) -> list[dict]:
    """Only messages enter inference; provenance and any other fields stay local."""
    messages = [{"role": m["role"], "content": m["content"]} for m in card["messages"]]
    # BioMistral's published Mistral template accepts user/assistant roles only.
    if model == "biomistral-7b":
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        messages = [m for m in messages if m["role"] != "system"]
        if system:
            messages[0]["content"] = system + "\n\n" + messages[0]["content"]
    return messages


def extract_answer(raw: str) -> tuple[str, object | None]:
    # Some Qwen templates put the opening <think> in the prompt, so only the
    # closing tag appears in the generated continuation. Preserve raw separately.
    if "</think>" in raw:
        answer = raw.rsplit("</think>", 1)[1].strip()
    elif "<think>" in raw:
        return "", None  # Generation ended before its final answer.
    else:
        answer = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", answer)
    if fenced:
        answer = fenced.group(1).strip()
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        parsed = None
    return answer, parsed


def api_runner(args, model_id: str):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env", override=False)
    if args.model in OPENAI_MODELS:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY in your environment or .env.")
        # Keep the selected endpoint explicit rather than using OPENAI_BASE_URL.
        base_url = args.base_url or "https://api.openai.com/v1"
    else:
        if not args.base_url:
            raise ValueError("Open-weight API inference needs --base-url.")
        base_url = args.base_url
        key = os.environ.get("INFERENCE_API_KEY", "local")
    client = OpenAI(api_key=key, base_url=base_url, timeout=args.timeout, max_retries=0)

    def generate(messages: list[dict], max_new_tokens: int) -> dict:
        if args.model in OPENAI_MODELS:
            response = client.responses.create(
                model=model_id, input=messages, store=False,
                **({"reasoning": {"effort": args.reasoning_effort}} if args.reasoning_effort else {}),
                max_output_tokens=max_new_tokens,
                service_tier="default",
            )
            refusals = [part.refusal for item in response.output
                        if item.type == "message" for part in item.content
                        if part.type == "refusal"]
            return {
                "raw_output": response.output_text,
                "returned_model": response.model,
                "response_id": response.id,
                "service_tier": response.service_tier,
                "returned_reasoning": response.reasoning.model_dump() if response.reasoning else None,
                "status": "refused" if refusals else response.status,
                "refusal": "\n".join(refusals) or None,
                "finish_reason": (response.incomplete_details.reason
                                  if response.incomplete_details else response.status),
                "usage": response.usage.model_dump() if response.usage else None,
            }
        extra = {}
        if args.model == "qwen3-8b":
            extra["chat_template_kwargs"] = {"enable_thinking": args.thinking}
        response = client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            **({"extra_body": extra} if extra else {}),
        )
        choice = response.choices[0]
        message = choice.message.model_dump()
        return {
            "raw_output": choice.message.content or "",
            "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
            "returned_model": response.model,
            "response_id": response.id,
            "status": "completed" if choice.finish_reason == "stop" else "incomplete",
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    return generate


def local_runner(args, model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "Local inference needs torch>=2.6, transformers>=4.51.3,<5, and accelerate. "
            "See README.md."
        ) from exc

    kwargs = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map=args.device, **kwargs,
    ).eval()
    # Read the actual model configuration; never shorten the scientific packet.
    context_limit = getattr(model.config, "max_position_embeddings", None)
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = eos_ids or [tokenizer.eos_token_id]

    def generate(messages: list[dict], max_new_tokens: int) -> dict:
        template_kwargs = {"enable_thinking": args.thinking} if args.model == "qwen3-8b" else {}
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **template_kwargs,
        ).to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        if context_limit and prompt_tokens + max_new_tokens > context_limit:
            raise ValueError(
                f"{prompt_tokens} prompt tokens + {max_new_tokens} output tokens "
                f"exceed this model's {context_limit}-token context. Reduce "
                "--max-new-tokens or use a model with more context; input was not truncated."
            )
        sampling = {"do_sample": args.temperature > 0}
        if args.temperature > 0:
            sampling.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **sampling,
            )[0, prompt_tokens:]
        stopped = len(generated) > 0 and generated[-1].item() in eos_ids
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        # Thinking-enabled Qwen starts its reasoning in the prompt. An unclosed
        # continuation contains no final answer even if it happens to look like JSON.
        unfinished_thinking = (args.model == "qwen3-8b" and args.thinking
                               and "</think>" not in raw)
        return {
            "raw_output": raw,
            "returned_model": model_id,
            "model_revision": getattr(model.config, "_commit_hash", args.revision),
            "status": "completed" if stopped else "incomplete",
            "finish_reason": "stop" if stopped else "length",
            "unfinished_thinking": unfinished_thinking,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": len(generated)},
        }

    return generate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="gpt-5.6-luna")
    parser.add_argument("--model-id", help="Override the API name or Hugging Face ID/local path.")
    parser.add_argument("--backend", choices=("auto", "api", "local"), default="auto",
                        help="Auto uses API for GPT or --base-url, otherwise local Transformers.")
    parser.add_argument("--base-url", help="OpenAI-compatible server URL, including /v1.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prompts/test")
    parser.add_argument("--dataset", choices=("all", "papers"), default="papers",
                        help="Both choices select the current paper test set.")
    parser.add_argument("--card-id", help="Run one exact test task ID.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many selected cards.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cards (default: all).")
    parser.add_argument("--output", type=Path, help="New JSONL file; defaults to inference_results/<model>-<time>.jsonl.")
    parser.add_argument("--resume", type=Path, nargs="+", default=[], metavar="JSONL",
                        help="Copy completed results from these files into a new output and run the remaining task IDs.")
    parser.add_argument("--max-new-tokens", type=int,
                        help="Override each task's saved generation budget (default: use the test manifest).")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"),
                        help="Optional OpenAI override; omitted by default to use the API default.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Open-weight models only; 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Open-weight models only.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Qwen3 only. API mode requires server chat_template_kwargs support.")
    parser.add_argument("--device", default="auto", help="Local model placement: auto, cpu, cuda:0, mps, etc.")
    parser.add_argument("--revision", help="Local Hugging Face revision (otherwise repository default).")
    parser.add_argument("--timeout", type=float, default=180.0, help="API request timeout in seconds.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Maximum in-flight API requests; refill each free slot immediately. Local models use 1.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect card selection without credentials, downloads, or calls.")
    args = parser.parse_args(argv)
    if (args.offset < 0 or (args.limit is not None and args.limit < 1)
            or (args.max_new_tokens is not None and args.max_new_tokens < 1)):
        parser.error("--offset must be nonnegative; --limit and --max-new-tokens must be positive.")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.timeout <= 0:
        parser.error("Require temperature >= 0, 0 < top-p <= 1, and timeout > 0.")
    if args.backend == "auto":
        args.backend = "api" if args.model in OPENAI_MODELS or args.base_url else "local"
    if args.backend == "local" and (args.model in OPENAI_MODELS or args.base_url):
        parser.error("Local inference requires an open-weight model and no --base-url.")
    if args.backend == "api" and args.model not in OPENAI_MODELS and not args.base_url:
        parser.error("Open-weight API inference requires --base-url.")
    if args.concurrency < 1 or (args.backend == "local" and args.concurrency != 1):
        parser.error("--concurrency must be positive and must be 1 for local inference.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cards = read_cards(args.data_dir, args.dataset)
    if args.card_id:
        cards = [c for c in cards if c["card_id"] == args.card_id]
    cards = cards[args.offset:]
    if args.limit is not None:
        cards = cards[:args.limit]
    if not cards:
        raise ValueError("No cards match this selection.")
    if len({card["card_id"] for card in cards}) != len(cards):
        raise ValueError("Selected input contains duplicate card_id values.")
    budgets = [args.max_new_tokens or card["generation_budget_tokens"] for card in cards]
    model_id = args.model_id or MODELS[args.model]
    completed = read_completed(args.resume, cards, model_id, args.backend)
    pending = [(i, card) for i, card in enumerate(cards) if card["card_id"] not in completed]
    summary = {"model": model_id, "backend": args.backend, "cards": len(cards),
               "datasets": dict(Counter(c["dataset"] for c in cards)),
               "partitions": dict(Counter(c["provenance"]["partition"] for c in cards)),
               "source_families": len({c["provenance"]["family_id"] for c in cards}),
               "generation_budget_tokens": dict(sorted(Counter(budgets).items())),
               "first_card_id": cards[0]["card_id"], "last_card_id": cards[-1]["card_id"],
               "resumed_completed": len(completed), "pending_calls": len(pending),
               "resume_sources": [str(path.resolve()) for path in args.resume]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or ROOT / "inference_results" / f"{args.model}-{now}.jsonl"
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new --output path.")
    generate = None
    if pending:
        generate = api_runner(args, model_id) if args.backend == "api" else local_runner(args, model_id)
    settings = {"generation_budget_source": "command_line" if args.max_new_tokens else "test_manifest"}
    if args.model in OPENAI_MODELS:
        if args.reasoning_effort:
            settings["reasoning_effort"] = args.reasoning_effort
        settings["service_tier"] = "default"
    else:
        settings.update(temperature=args.temperature, top_p=args.top_p)
        if args.model == "qwen3-8b":
            settings["thinking"] = args.thinking
    output.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.monotonic()

    def run_card(index, card):
        started = time.monotonic()
        result = {"card_id": card["card_id"], "dataset": card["dataset"],
                  "task_family": card["task_family"], "provenance": card["provenance"],
                  "selection_index": index, "requested_model": model_id, "backend": args.backend,
                  "settings": {**settings, "max_new_tokens": budgets[index]},
                  "created_at": datetime.now(timezone.utc).isoformat()}
        try:
            result.update(generate(model_messages(card, args.model), budgets[index]))
            answer, parsed = extract_answer(result["raw_output"])
            if result.get("unfinished_thinking"):
                answer, parsed = "", None
            result.update(answer_text=answer, parsed_answer=parsed)
            if result["status"] == "completed" and not answer:
                result["status"] = "empty"
        except Exception as exc:
            # Record request metadata without echoing credentials or request bodies.
            error = type(exc).__name__
            if getattr(exc, "status_code", None):
                error += f": HTTP {exc.status_code}; request_id={getattr(exc, 'request_id', None)}"
            result.update(status="error", error=error)
        result["cost"] = estimate_cost(result)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return result

    results = []
    new_results = []
    with output.open("x", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for index, card in enumerate(cards):
            if card["card_id"] in completed:
                result = {**completed[card["card_id"]], "selection_index": index}
                results.append(result)
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
        sink.flush()
        if completed:
            print(f"Reused {len(completed)} completed results; {len(pending)} tasks remain.", flush=True)

        remaining = iter(pending)
        active = set()

        def fill_slots():
            while len(active) < args.concurrency:
                item = next(remaining, None)
                if item is None:
                    break
                active.add(pool.submit(run_card, *item))

        fill_slots()
        stopped_on_error = False
        while active:
            finished, active = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                results.append(result)
                new_results.append(result)
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"[{len(results)}/{len(cards)}] {result['card_id']}: {result['status']}", flush=True)
                if result["status"] == "error":
                    stopped_on_error = True
                    print(result["error"], file=sys.stderr)
            if not stopped_on_error:
                fill_slots()
            # After an error, drain and save in-flight calls without submitting more.
    costed = [r for r in results if r.get("cost") is not None]
    summary.update(
        results_file=str(output.resolve()), recorded_calls=len(new_results),
        result_rows=len(results), unattempted_cards=len(cards) - len(results),
        stopped_on_error=stopped_on_error,
        statuses=dict(Counter(r["status"] for r in results)),
        new_call_statuses=dict(Counter(r["status"] for r in new_results)),
        parsed_json_answers=sum(r.get("parsed_answer") is not None for r in results),
        usage={key: sum((r.get("usage") or {}).get(key, 0) for r in results)
               for key in ("input_tokens", "output_tokens", "total_tokens")},
        estimated_cost_usd=round(sum(r["cost"]["estimated_usd"] for r in costed), 10) if costed else None,
        calls_with_cost_estimate=len(costed), calls_without_cost_estimate=len(results) - len(costed),
        concurrency=args.concurrency, timeout_seconds=args.timeout, settings=settings,
        elapsed_wall_seconds=round(time.monotonic() - run_started, 3),
        accounting_note="Usage and cost totals cover all output rows, including reused results; recorded_calls and elapsed_wall_seconds cover this invocation only. Reused rows retain their original generation settings and timings.",
        cost_note="Published-rate estimate from returned usage, not an invoiced charge. No SDK retries; error attempts may have unreturned usage.",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {output.resolve()}")
    print(f"Summary {summary_path.resolve()}")
    return 1 if any(r["status"] != "completed" for r in results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
