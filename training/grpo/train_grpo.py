#!/usr/bin/env python3
"""GRPO from SFT v2 step-125 (docs/specs/metrics.md §10). TRL 0.25.1.

The step-125 LoRA adapter itself is trained further. Four rollouts per card, rewards from reward.py
(one Luna call per valid rollout), advantages not divided by the group standard deviation, no KL
term, DAPO token-level loss, truncated rollouts left out of the loss.

TRL colocate mode: every GPU holds its own vLLM copy (35% of its memory), generates its share of the
rollouts and trains on it (data parallel). Generation is short next to training, so both GPUs train.

vLLM applies the current LoRA at runtime, as in SFT v2 selection. TRL's own sync merges the LoRA into
the bf16 base weights; step-125's LoRA changes are about 1e-4 per weight, and the bf16 merge loses
30-60% of them.

vLLM's prefix caching is off. With it, a card's four identical prompts share cached prompt blocks,
and vLLM 0.10.2 then generated answers that loop until the length limit (smoke runs 2026-10-05:
valid .31-.34 against step-125's .85). step-125 alone in vLLM, 8 development and 8 training cards,
4 rollouts each, TRL's vLLM settings: 0 of 64 at the 4,096-token limit with prefix caching off,
19 of 64 with it on.

    torchrun --nproc_per_node 2 training/grpo/train_grpo.py --output_dir outputs/grpo-001
    torchrun --nproc_per_node 2 training/grpo/train_grpo.py --output_dir outputs/smoke --max_steps 3
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BUNDLE / "metrics"))

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EOS_ID = 151645  # <|im_end|>
PAD_ID = 151643  # <|endoftext|>


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_file", type=Path, default=BUNDLE / "data/training/grpo-core-train.jsonl.gz")
    parser.add_argument("--adapter", type=Path, default=BUNDLE / "outputs/sft/checkpoint-150")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume_from_checkpoint", type=Path,
                        help="Restore the adapter, optimizer, scheduler, RNG and trainer state.")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop early (smoke runs).")
    parser.add_argument("--cards_per_step", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--micro_batch", type=int, default=2, help="Rollouts per forward/backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--judge_workers", type=int, default=64)
    parser.add_argument("--reward_mode", choices=("legacy", "content", "core"), default="legacy",
                        help="content: historical C-F; core: frozen core-content-v3; neither uses an L0 gate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35,
                        help="vLLM's share of each GPU (weights 16 GB, the rest KV cache).")
    parser.add_argument("--max_prompt_length", type=int, default=12800,
                        help="Longest training prompt is 12.3k tokens; sets vLLM's context with the completion.")
    args = parser.parse_args()
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rollouts = args.cards_per_step * args.num_generations
    if rollouts % (args.micro_batch * args.world_size):
        parser.error("cards_per_step * num_generations must be divisible by micro_batch * GPUs.")
    args.gradient_accumulation_steps = rollouts // (args.micro_batch * args.world_size)
    return args


def main():
    args = parse_args()
    main_process = args.local_rank == 0
    if main_process and args.output_dir.exists() and args.resume_from_checkpoint is None:
        raise RuntimeError(f"{args.output_dir} exists; use a new output directory.")
    if args.resume_from_checkpoint is not None:
        assert args.resume_from_checkpoint.parent.resolve() == args.output_dir.resolve()
    from dotenv import load_dotenv
    load_dotenv(BUNDLE / ".env", override=False)  # OPENAI_API_KEY for the L3 judge
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"Put OPENAI_API_KEY in {BUNDLE / '.env'} (the L3 judge needs it).")
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import trl.trainer.grpo_trainer as grpo_module
    from vllm.lora.request import LoRARequest
    if args.reward_mode == "core":
        sys.path.insert(0, str(BUNDLE / "training/grpo"))
        from core_reward import Reward
    elif args.reward_mode == "content":
        sys.path.insert(0, str(BUNDLE / "training/grpo"))
        from content_reward import Reward
    else:
        from reward import Reward

    # vLLM keeps the pinned base weights, applies the LoRA at runtime and caches no prompts (module docstring).
    grpo_module.LLM = partial(grpo_module.LLM, revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
                              enable_lora=True, max_lora_rank=16, max_loras=1, max_cpu_loras=1,
                              enable_prefix_caching=False)

    class Trainer(GRPOTrainer):
        def _move_model_to_vllm(self):
            """Instead of merging: save the current adapter and generate with it as a new LoRA request."""
            self.lora_id = getattr(self, "lora_id", 0) + 1
            path = Path(f"/tmp/hypomap-vllm-lora/rank{self.accelerator.process_index}")
            self.model.save_pretrained(str(path))
            self.lora_request = LoRARequest(f"policy-{self.lora_id}", self.lora_id, str(path))
    with gzip.open(args.train_file, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    dataset = Dataset.from_list(rows)
    d1 = sum(bool(json.loads(r["card"])["d1"]) for r in rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    assert tokenizer.eos_token_id == EOS_ID and tokenizer.pad_token_id == PAD_ID
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": args.local_rank})
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(args.resume_from_checkpoint or args.adapter), is_trainable=True)
    model.enable_input_require_grads()  # without it, gradient checkpointing leaves the adapter with no gradient

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = GRPOConfig(
        output_dir=str(args.output_dir), seed=args.seed,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, lr_scheduler_type="constant_with_warmup", warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations, max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length, temperature=args.temperature, top_p=args.top_p,
        beta=args.beta, scale_rewards="none", loss_type="dapo", mask_truncated_completions=True,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=False,  # sleep drops the base weights, which the runtime LoRA needs
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=args.save_steps, save_total_limit=None,
        report_to="none", remove_unused_columns=False,
        ddp_find_unused_parameters=False,  # every LoRA weight is used; the check costs a graph traversal per step
    )
    # Each process scores its own rollouts; one log per process.
    log = "rewards.jsonl" if args.world_size == 1 else f"rewards-rank{args.local_rank}.jsonl"
    # vLLM also stops at <|endoftext|> (Qwen3 generation config); TRL and selection count both as finished.
    reward = Reward(args.output_dir / log, eos_ids=(EOS_ID, PAD_ID), workers=args.judge_workers)
    trainer = Trainer(model=model, reward_funcs=reward, args=config, train_dataset=dataset,
                      processing_class=tokenizer)
    generate = trainer.llm.generate  # TRL calls llm.generate(prompts, sampling_params=..., use_tqdm=False)
    trainer.llm.generate = lambda *a, **k: generate(*a, lora_request=trainer.lora_request, **k)

    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "packages": {p: version(p) for p in ("torch", "transformers", "trl", "peft", "vllm", "openai")},
        "model": MODEL_ID, "revision": MODEL_REVISION, "start_adapter": str(args.adapter),
        "train_cards": len(rows), "train_d1_cards": d1,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "reward": (reward.description if args.reward_mode == "core" else
                   "C - F: raw-response requirement coverage minus forbidden-inference rate; no format gate"
                   if args.reward_mode == "content" else
                   "docs/specs/metrics.md §10: invalid 0; D1 cards (2·D1 + L2 + L3)/4; other cards (L2 + L3)/2"),
    }
    meta_path = args.output_dir / "run_metadata.json"
    if main_process:
        if args.resume_from_checkpoint is not None:
            old = json.loads(meta_path.read_text())
            metadata["started_at"] = old["started_at"]
            state = json.loads((args.resume_from_checkpoint / "trainer_state.json").read_text())
            metadata["resume_history"] = old.get("resume_history", []) + [{
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(args.resume_from_checkpoint), "global_step": state["global_step"]}]
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: metadata[k] for k in ("train_cards", "train_d1_cards", "settings")}, indent=2), flush=True)

    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_state()
    trainer.save_model(str(args.output_dir / "final"))
    if main_process:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), global_step=trainer.state.global_step,
                        checkpoints=sorted(p.name for p in args.output_dir.glob("checkpoint-*")))
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("Training complete:", ", ".join(metadata["checkpoints"]), flush=True)


if __name__ == "__main__":
    main()
