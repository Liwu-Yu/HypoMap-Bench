# Reproducing the HypoMap-Bench experiments

Run these commands from the repository root using Python 3.12. Install the scoring and analysis dependencies with `python -m pip install -r requirements.txt`.

API keys, only for steps that call a model: `OPENROUTER_API_KEY` (Gemini 3.8 Flash judges),
`OPENAI_API_KEY` (GPT-6 Luna answers and the RL reward judge). Put them in the environment or in `.env` here.

## Where each part of the paper lives

| Paper | Files |
| --- | --- |
| §3.1 task format: hypothesis map, request, response contract, the six operations | `hypomap/tasks.py` (schemas, contract), `hypomap/grading.py` (answer validation, patch application $M\oplus\Delta$) |
| §3.2 construction and quality control (appendices *Benchmark Details* and *Prompts*) | `prompts/`: full instructions of the author, leak checker, reviewer, blind solver, source auditor and the minimal-pair (variant) author and reviewer |
| §3.2 statistics and splits (task-family table, data-statistics table) | `data/tasks/`, `data/papers.jsonl` (see `data/README.md`) |
| §3.3 validity and scope | `metrics/score_runs.py` (L0 = validity, L2 = scope via `metrics/reward.py` `l2`) |
| §3.3 decision (balanced accuracy, act recall, over-edit, pair accuracy) | `metrics/score_d1.py` (the decision rule), `metrics/score_runs.py` (L1) |
| §3.3 correctness (closed check, Gemini 3.8 Flash) | `metrics/l3_judge.py` (prompt, items, parsing), called by `metrics/score_runs.py` (L3) |
| §3.3 map quality (side-by-side judge) | `metrics/map_quality.py`; maps rendered by `metrics/render_workspace.py`; rubric in `metrics/l4_judge.py` `DIMENSIONS` |
| Judge-validation table (§3.3) | correctness: `metrics/l3_check.py`; expert alignment: `metrics/map_quality.py alignment` and `metrics/l4_calibrate.py` (method only; the expert data are not released) |
| §4 models; appendix *Experimental Settings* | `inference.py` (OpenAI Responses API, any OpenAI-compatible server such as vLLM, or local Transformers) |
| §4 SFT; appendices *Experimental Settings*, *Training Algorithms* | `training/sft/train.py`, `training/sft/requirements.txt` |
| §4 RL (GRPO, reward Eq. 1); same appendices | `training/grpo/train_grpo.py`; reward judge `training/grpo/core_reward.py` → `content_reward_v2.py` (prompt, $D$, $A$, $C_{aux}$, $E$, Eq. 1) → `content_reward.py` |
| §5 main results table and family figure; appendix *Full Test Results* (intervals, paired differences, per family, decision per family, validity components) | `analysis/pooled.py`, `analysis/paper_tables.py` |
| Appendix table *What RL moves* | `analysis/controls.py`; the earlier validity-gated reward is `metrics/reward.py` |

Names in code and results: L0 validity, L1 decision, L2 scope, L3 correctness; map quality is the side-by-side
gain of `metrics/map_quality.py` (the `l4*` columns of `score_runs.py` belong to an absolute judge the paper does
not use). Models: `gpt-6-luna`, `Qwen/Qwen3-8B`, `May2222/Fisher-R1-14B`, `BioMistral/BioMistral-7B`, `sft-v1`
(SFT w/o pairs), `sft-150` (SFT), `core-grpo-40` (SFT+RL), `core-grpo-200` (SFT+RL continued to 200 steps). A
"D1 card" is a decision card; `act` / `no-act` are its labels. Some comments cite internal design notes
(`docs/specs/...`) that are not part of this release; the paper describes the same rules.

## Reproduce the paper's numbers from the released files (no GPU, no API calls)

```bash
python analysis/pooled.py && python analysis/paper_tables.py    # main results, family figure source, appendix tables -> analysis/out/
python analysis/controls.py                                      # "What RL moves" -> analysis/out/tables/rl-controls.tex
python metrics/l3_check.py report --out data/results/judge-checks --judge gemini-3.8-flash   # judge validation: correctness
python metrics/map_quality.py score --judge gemini               # map quality per model from the cached judge calls
python metrics/map_quality.py report                             # reference update better than the original: 73%
```

Re-score all test answers from scratch; every correctness verdict is cached, so no judge is called:

```bash
mkdir -p /tmp/rescore && cp data/results/core-grpo200/l3-gemini-3.8-flash.jsonl /tmp/rescore/
python metrics/score_runs.py data/outputs/test/{Qwen3-8B,sft-v1,gpt-6-luna,Fisher-R1-14B,BioMistral-7B,sft-150,core-grpo-40,core-grpo-200}.jsonl \
  --split test --baseline sft-150 --out /tmp/rescore    # = data/results/core-grpo200/{rows,summary}
```

`data/results/final-seven/` holds the run behind the main results table (one SFT-150 task has no correctness verdict there,
because the judge used up its output budget six times; the later run in `core-grpo200/` has it).

## Evaluate a model on HypoMap-Bench

1. Answers. Serve the model with an OpenAI-compatible server (we used vLLM 0.10.2) or use an API model. Prompts and per-task token budgets are in `data/prompts/{test,development}/papers.jsonl`.

   ```bash
   vllm serve Qwen/Qwen3-8B --max-model-len 32768
   python inference.py --model qwen3-8b --base-url http://127.0.0.1:8000/v1 --data-dir data/prompts/test \
     --concurrency 32 --output answers/qwen3-8b.jsonl
   python inference.py --model gpt-6-luna --data-dir data/prompts/test --concurrency 16 --output answers/luna.jsonl
   ```

   Presets (`--model`): `gpt-6-luna`, `qwen3-8b`, `fisher-r1-14b`, `biomistral-7b`; `--model-id` names any other
   model. Sampling for open models: temperature 0.6, top-p 0.95 (appendix *Experimental Settings*).

2. Validity, decision, scope and correctness (one Gemini call per answer):

   ```bash
   python metrics/score_runs.py answers/*.jsonl --split test --baseline MODEL --out results/
   ```

3. Map quality: render each updated map, judge every changed map against its original in both orders, score.

   ```bash
   python metrics/score_runs.py answers/*.jsonl --split test --l3-judges --hide-merged --out mq/workspaces
   python metrics/map_quality.py run --judge gemini --models --dir mq
   python metrics/map_quality.py score --judge gemini --dir mq
   ```

## Training

SFT (one A100 80GB, about 3.5 h). The token sequences in `data/training/` are the exact inputs; the
script checks the prompt mask, the single supervised end-of-turn token and the lengths before training.

```bash
pip install -r training/sft/requirements.txt       # torch 2.6.0, transformers 4.51.3, peft 0.15.2, flash-attn 2.7.4
python training/sft/train.py --epochs 2 --learning_rate 1e-4 --warmup_ratio 0.03 --train_batch_size 1 \
  --gradient_accumulation_steps 16 --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --eval_steps 25 \
  --seed 42 --output_dir outputs/sft
```

The checkpoint is selected on generated development answers (content reward, appendix *Training Algorithms*), not on loss: answer
`data/prompts/development` with each saved adapter and score them with the reward judge.

GRPO (two A100 80GB, about 6 h; torch 2.8.0, vLLM 0.10.2, TRL 0.25.1, transformers 4.56.2, PEFT 0.17.1,
openai 1.109.1). It continues the selected SFT adapter; `data/training/grpo-core-train.jsonl.gz` holds the 640
task draws in training order (`grpo-sampling.json`: seed 42, half uniform over tasks, half over families).

```bash
torchrun --nproc_per_node 2 training/grpo/train_grpo.py --adapter outputs/sft/checkpoint-150 --reward_mode core \
  --train_file data/training/grpo-core-train.jsonl.gz --max_steps 40 --save_steps 20 --cards_per_step 16 \
  --num_generations 4 --micro_batch 2 --learning_rate 1e-5 --warmup_steps 5 --beta 0.0 --temperature 0.6 \
  --top_p 0.95 --max_prompt_length 12800 --max_completion_length 4096 --vllm_gpu_memory_utilization 0.35 \
  --judge_workers 32 --seed 42 --output_dir outputs/grpo
```

## Not included

- Full texts of the 722 source articles: all are CC BY open-access articles; `data/papers.jsonl` gives each
  paper's PMCID and DOI.
- The collection tooling and the per-paper construction records (agent transcripts, reviews, audits). The role
  instructions in `prompts/` describe each check; they refer to commands of that tooling.
- Trained LoRA adapters.
- The expert-alignment data: the 20 researcher-built maps (participants' own research plans from a separate
  study), their expert ratings and the judge's calls on them. The alignment code is included.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`data/LICENSE`); the source articles are CC BY, and model outputs are
also subject to the terms of the models that produced them.
