#!/usr/bin/env python3
"""Qwen3-8B LoRA SFT on HypoMap-Bench's saved token sequences; one CUDA GPU."""

import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EOS_ID = 151645  # <|im_end|> in the pinned Qwen tokenizer.
PAD_ID = 151643  # <|endoftext|>; never substitute EOS when masking labels.
MAX_LENGTH = 32768
TENSOR_KEYS = ("input_ids", "attention_mask", "labels")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pad_features(features):
    """Right-pad only; preserve every saved input, attention and label value."""
    width = max(len(row["input_ids"]) for row in features)
    padding = {"input_ids": PAD_ID, "attention_mask": 0, "labels": -100}
    return {
        key: [row[key] + [padding[key]] * (width - len(row[key])) for row in features]
        for key in TENSOR_KEYS
    }


def collate_batch(features):
    import torch
    return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_features(features).items()}


def load_lora_model(lora_config, seed):
    """The benchmark and full run use the same model, tokenizer and adapter setup."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This script requires one visible BF16-capable CUDA GPU. Set CUDA_VISIBLE_DEVICES=0.")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.eos_token_id != EOS_ID or tokenizer.pad_token_id != PAD_ID:
        raise ValueError("Loaded tokenizer special tokens differ from the saved dataset contract.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": 0},
    )
    model.config.use_cache = False
    return get_peft_model(model, LoraConfig(**lora_config)), tokenizer


def load_data(path, role):
    """Check saved supervision without re-tokenizing, cropping, or packing."""
    examples, lengths, supervised_counts = [], [], []
    task_ids, families = set(), set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            meta = row["split_meta"]
            actual_role = meta["train_role"] if meta["split"] == "train" else meta["split"]
            task_id = meta["task_id"]
            ids, labels, attention = (row[k] for k in ("input_ids", "labels", "attention_mask"))
            prompt = row["prompt_input_ids"]
            active = [i for i, label in enumerate(labels) if label != -100]
            error = None
            if actual_role != role or task_id in task_ids:
                error = "wrong split or duplicate task ID"
            elif not (0 < len(ids) == len(labels) == len(attention) <= MAX_LENGTH):
                error = "tensor lengths disagree or exceed native context"
            elif ids[:len(prompt)] != prompt or any(value != 1 for value in attention):
                error = "saved prompt or attention mask changed"
            elif not active or active[0] != len(prompt):
                error = "supervision must begin immediately after the masked prompt"
            elif labels.count(EOS_ID) != 1 or labels[active[-1]] != EOS_ID:
                error = "expected exactly one supervised EOS, at the end of the target"
            elif active != list(range(len(prompt), active[-1] + 1)):
                error = "assistant supervision contains gaps"
            elif any(labels[i] != ids[i] for i in active):
                error = "supervised labels differ from input tokens"
            elif ids[active[-1] + 1:] != [198]:
                error = "expected the saved, masked formatting newline after EOS"
            if error:
                raise ValueError(f"{path}:{line_number} ({task_id}): {error}")
            examples.append({key: row[key] for key in TENSOR_KEYS})
            lengths.append(len(ids))
            supervised_counts.append(len(active))
            task_ids.add(task_id)
            families.add(meta["family_id"])
    if not examples:
        raise ValueError(f"Empty dataset: {path}")
    ordered = sorted(lengths)
    summary = {
        "path": str(path.resolve()), "role": role, "examples": len(examples),
        "source_families": len(families), "sequence_tokens": sum(lengths),
        "supervised_tokens": sum(supervised_counts), "min_tokens": ordered[0],
        "median_tokens": ordered[len(ordered) // 2], "max_tokens": ordered[-1],
        "verified_supervised_eos": len(examples),
    }
    return examples, summary, task_ids, families


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, default=ROOT.parents[1] / "data/training/sft-tokens.jsonl.gz")
    parser.add_argument("--eval_file", type=Path, default=ROOT.parents[1] / "data/training/development-tokens.jsonl.gz")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--dry_run", action="store_true", help="Check data and write metadata; no model download or training.")
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--train_batch_size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=int, default=25, help="Optimizer steps between evaluation and checkpoints.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.gradient_accumulation_steps, args.eval_steps, args.lora_rank, args.lora_alpha) < 1:
        parser.error("Accumulation, eval interval, LoRA rank and alpha must be positive.")
    if args.learning_rate <= 0 or not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio < 1:
        parser.error("Use a positive learning rate and dropout/warmup ratios in [0, 1).")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = ROOT / "outputs" / (("dry-run-" if args.dry_run else "qwen3-8b-lora-") + stamp)
    return args


def package_versions():
    result = {}
    for name in ("torch", "transformers", "tokenizers", "peft", "accelerate", "flash-attn", "safetensors"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def main():
    args = parse_args()
    train, train_summary, train_ids, train_families = load_data(args.train_file, "sft")
    development, eval_summary, eval_ids, eval_families = load_data(args.eval_file, "development")
    if train_ids & eval_ids or train_families & eval_families:
        raise ValueError("Training and development must have separate task IDs and source families.")

    # Include each epoch's partial accumulation group. Transformers 4.51.3's
    # epoch-to-step estimate rounds down; an explicit total preserves both passes.
    batches_per_epoch = math.ceil(len(train) / args.train_batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_steps = args.epochs * steps_per_epoch
    training_config = {
        "num_train_epochs": args.epochs, "max_steps": total_steps,
        "per_device_train_batch_size": args.train_batch_size, "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate, "optim": "adamw_torch",
        "lr_scheduler_type": "cosine", "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.0, "max_grad_norm": 1.0,
        "bf16": True, "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": "steps", "eval_steps": args.eval_steps,
        "save_strategy": "steps", "save_steps": args.eval_steps,
        # Keep every checkpoint; selection uses generated development D1 answers
        # afterwards (docs/specs/decision-variants.md §8), not development loss.
        "save_total_limit": None, "load_best_model_at_end": False,
        "eval_on_start": True,
        "prediction_loss_only": True, "logging_steps": 1,
        "seed": args.seed, "data_seed": args.seed,
        "label_names": ["labels"], "remove_unused_columns": False,
        "dataloader_num_workers": 0, "dataloader_drop_last": False,
        "report_to": "none", "push_to_hub": False,
    }
    lora_config = {
        "task_type": "CAUSAL_LM", "target_modules": "all-linear",
        "r": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "bias": "none",
        "revision": MODEL_REVISION,
    }
    metadata = {
        "status": "dry_run_passed" if args.dry_run else "loading_model",
        "started_at": utc_now(), "command": sys.argv,
        "python": sys.version, "platform": platform.platform(),
        "package_versions": package_versions(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION, "eos_token_id": EOS_ID,
        "attention_implementation": "flash_attention_2", "dtype": "bfloat16",
        "packing": False, "truncation": False, "max_sequence_tokens": MAX_LENGTH,
        "dataset_snapshot": "decision-variants-v1", "split_manifest": "data/tasks",
        "datasets": {"train": train_summary, "development": eval_summary},
        "training_config": training_config, "lora_config": lora_config,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "planned_optimizer_steps": total_steps,
        "evaluation_steps": sorted(set(range(0, total_steps + 1, args.eval_steps)) | {total_steps}),  # 0 = eval_on_start
        "selection_metric": "Generated development D1 balanced accuracy over all D1 families (evaluated after training); eval_loss is logged only",
        "tensor_checks_passed": True, "gpu_execution_verified": False,
    }
    # A new directory keeps separate runs and their metadata easy to identify.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = args.output_dir / "run_metadata.json"

    def save_metadata(**updates):
        metadata.update(updates)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save_metadata()
    print(json.dumps({"datasets": metadata["datasets"], "optimizer_steps": total_steps,
                      "evaluation_steps": metadata["evaluation_steps"],
                      "run_metadata": str(metadata_path)}, indent=2), flush=True)
    if args.dry_run:
        print("Dry run passed. Model loading, CUDA execution and training were not performed.")
        return

    # Heavy dependencies are deliberately below --dry_run.
    import torch
    from transformers import (
        Trainer, TrainerCallback, TrainingArguments,
    )

    model, tokenizer = load_lora_model(lora_config, args.seed)
    device = torch.cuda.get_device_properties(0)
    save_metadata(gpu={"name": device.name, "total_memory_bytes": device.total_memory,
                       "cuda_version": torch.version.cuda, "visible_device_count": 1})
    model.print_trainable_parameters()
    trainable, total = model.get_nb_trainable_parameters()
    save_metadata(status="training", trainable_parameters=trainable, total_parameters=total,
                  adapted_modules=[name for name, module in model.named_modules() if hasattr(module, "lora_A")])

    class SaveProgress(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Evaluate and save the final partial interval too.
            if state.global_step >= state.max_steps:
                control.should_evaluate = control.should_save = control.should_log = True
            return control

        def on_save(self, args, state, control, **kwargs):
            save_metadata(global_step=state.global_step, epoch=state.epoch,
                          last_checkpoint=f"checkpoint-{state.global_step}",
                          last_checkpoint_at=utc_now(), gpu_execution_verified=True)

    trainer = Trainer(
        model=model, args=TrainingArguments(output_dir=str(args.output_dir), **training_config),
        train_dataset=train, eval_dataset=development, processing_class=tokenizer,
        # Padding never changes the saved prompt/assistant labels or adds a target EOS.
        data_collator=collate_batch, callbacks=[SaveProgress()],
    )
    result = trainer.train()
    trainer.save_state()
    eval_losses = {row["step"]: row["eval_loss"]
                   for row in trainer.state.log_history if "eval_loss" in row}
    checkpoints = sorted(path.name for path in args.output_dir.glob("checkpoint-*"))
    save_metadata(
        status="completed", finished_at=utc_now(), global_step=trainer.state.global_step,
        epoch=trainer.state.epoch, checkpoints=checkpoints, eval_loss_by_step=eval_losses,
        train_metrics=result.metrics, gpu_execution_verified=True,
        peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    print(f"Training complete. Checkpoints for D1 selection: {', '.join(checkpoints)}", flush=True)


if __name__ == "__main__":
    main()
