"""TRL adapter around the frozen, already calibrated core-content-v3 judge."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics as st
import sys
import time

from openai import APIError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/grpo"))
from content_reward import diagnostics
from content_reward_v2 import FORMULA, SYSTEM, VERSION, judge

JUDGE_ATTEMPTS = 10  # backoff 5, 10, 20, 40, then 60 s


class Reward:
    __name__ = "core_content_v3"
    description = f"{VERSION}: {FORMULA}; final text when available; no format gate"

    def __init__(self, log_path, eos_ids, workers=32):
        from openai import OpenAI
        self.client = OpenAI(base_url="https://api.openai.com/v1", max_retries=8, timeout=300)
        self.log_path, self.eos_ids, self.workers = Path(log_path), set(eos_ids), workers
        if self.log_path.name in ("rewards.jsonl", "rewards-rank0.jsonl"):
            (self.log_path.parent / "reward-prompt.txt").write_text(SYSTEM)
            (self.log_path.parent / "reward-definition.txt").write_text(self.description + "\n")

    def __call__(self, prompts, completions, completion_ids, card, trainer_state=None, **kwargs):
        def call(item):
            payload, text, ids = item
            parsed = json.loads(payload)
            # Retry the same call on any API error (including a spurious 400 such as
            # bio_policy, which the SDK never retries) or an exhausted parse retry.
            for attempt in range(JUDGE_ATTEMPTS):
                try:
                    scored = judge(self.client, parsed, text)
                    break
                except (APIError, RuntimeError) as exc:
                    if attempt == JUDGE_ATTEMPTS - 1:
                        raise
                    print(f"judge retry {attempt + 1}/{JUDGE_ATTEMPTS - 1} {parsed['task_id']}: "
                          f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    time.sleep(min(60, 5 * 2 ** attempt))
            return {**scored, **diagnostics(parsed, text, bool(ids) and ids[-1] in self.eos_ids)}

        started = time.monotonic()
        with ThreadPoolExecutor(self.workers) as pool:
            rows = list(pool.map(call, zip(card, completions, completion_ids)))
        step = trainer_state.global_step if trainer_state is not None else None
        with self.log_path.open("a", encoding="utf-8") as sink:
            for row, text, ids in zip(rows, completions, completion_ids):
                sink.write(json.dumps({"step": step, **row, "tokens": len(ids), "completion": text},
                                      ensure_ascii=False) + "\n")
        print(json.dumps({"step": step, "reward_version": VERSION, "n": len(rows),
                          "judge_seconds": round(time.monotonic() - started, 1),
                          **{k: st.mean(r[k] for r in rows) for k in ("D", "A", "C_aux", "E", "reward")}}),
              flush=True)
        return [row["reward"] for row in rows]
