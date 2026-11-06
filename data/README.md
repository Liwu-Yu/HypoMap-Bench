# HypoMap-Bench data

This directory contains the English-language benchmark tasks, paper identifiers, model outputs, and cached evaluations used in the paper.

## Tasks (`tasks/`)

| File | Tasks | Papers | Decision cards (act / no-act) | Minimal-pair variants |
| --- | ---: | ---: | --- | ---: |
| `sft.jsonl.gz` | 1,771 | 468 | 424 (210 / 214) | 105 |
| `development.jsonl.gz` | 217 | 49 | 56 (28 / 28) | 16 |
| `test.jsonl.gz` | 479 | 122 | 108 (52 / 56) | 28 |
| `rl.jsonl.gz` | 789 | 274 | 373 (114 / 259) | 111 |

3,178 distinct tasks from 722 papers: the splits are disjoint by paper, except that 191 RL papers and 78 RL tasks
also appear in SFT. `rl.jsonl.gz` covers six of the nine families (local repair, alternative explanation and
redundancy are held out from RL), in the order that `training/grpo-sampling.json` indexes.

One JSON object per line:

- `split_meta`: `task_id`, `paper_id`, `family_id` (source family; the split unit), `domain` (`biomedical` or
  `materials_sensors`), `task_family` (one of the nine families), `answer_mode`, `d1` (`act`, `no-act`, or null
  for a task that is not a decision card), `is_variant` and `parent_task_id` (a minimal-pair variant and the task
  it was built from), `generation_budget_tokens` (8,192 or 16,384; null for RL variants, which were never
  answered outside training), and bookkeeping fields of the split assignment (`split`, `train_role`, `grpo_tier`,
  `v1_task_id`, `new_test_card`, `modified_test_card`).
- `messages`: the system and user messages every model received; the user message holds the public task packet.
- `input`: the public task packet as JSON: `objective` (the root $h$), `current_map` (aspects as nodes and their
  relations), `experiment_plans`, `visible_observations` (with the aspects each bears on), `existing_interpretations`,
  `constraints`, `background_packet`, `question`, `user_guidance` and `response_contract` (allowed operations,
  required fields, patch rules). Objects carry `source_spans` (passage IDs of the source article),
  `evidence_basis` (`reported` or `inferred`) and, if inferred, `inference_reason`.
- `output`: the reviewed reference answer (`aspect_assessments`, `findings`, `proposed_updates`,
  `verification_needs`, `preserved_ids`, `information_requests`, `no_change_reason`).
- `reasoning_demonstration`: the reasoning written from the public input; the SFT target puts it in the thinking
  block before the reference answer.
- `items`: the private reviewed items the correctness judge reads: `answer_requirements` (what a correct answer
  must establish), `forbidden_inferences` (conclusions the visible evidence does not license) and
  `acceptable_alternatives`.

Use only `messages` (or `input`) as model input; `output`, `reasoning_demonstration` and `items` are answers.

`papers.jsonl`: each paper's `paper_id`, PMCID, DOI, domain and the splits that use it. The articles were
retrieved through Europe PMC and are open access under CC BY licenses; their full texts are not redistributed here.

## Model inputs and training files

- `prompts/{test,development}/papers.jsonl`: the same tasks in the input format of `inference.py` (`card_id`,
  `messages`, `generation_budget_tokens`, `task_family`, `provenance`).
- `training/sft-tokens.jsonl.gz`, `training/development-tokens.jsonl.gz`: the Qwen3-8B token sequences that
  `training/sft/train.py` reads: `input_ids`, `labels` (-100 on the prompt), `attention_mask`, `prompt_input_ids`.
- `training/grpo-core-train.jsonl.gz`: the 640 GRPO task draws in training order (rendered prompt plus the task
  and its reviewed items); `training/grpo-sampling.json`: how they were drawn (seed 42).

## Model outputs and scores (the paper's results)

- `outputs/test/*.jsonl`: one answer per test task for each model, as written by `inference.py`
  (`raw_output`, `answer_text`, `parsed_answer`, `finish_reason`, `usage`, settings). `Qwen3-8B`, `gpt-6-luna`,
  `Fisher-R1-14B`, `BioMistral-7B`; `sft-v1` = SFT w/o pairs, `sft-150` = SFT, `core-grpo-40` = SFT+RL,
  `core-grpo-200` = SFT+RL continued to 200 steps.
- `results/final-seven/`: `metrics/score_runs.py` on the seven models of the main results table (`rows.jsonl`
  one row per model and task, `summary.json` per family, `pooled.json` from `analysis/pooled.py`) and every
  correctness call (`l3-gemini-3.8-flash.jsonl`: the judge's item verdicts with quotes and reasons).
- `results/core-grpo200/`: the same with the 200-step RL run added.
- `results/map-quality/`: the side-by-side map-quality judge on test: the rendered maps
  (`workspaces/l4-workspaces.jsonl`), every judge call (`calls-gemini.jsonl`), the gain per model and task
  (`task-gains-gemini.jsonl`) and pooled scores (`scores-gemini.json`).
- `results/judge-checks/`: the correctness-judge validation on the 217 development tasks: reference answers,
  answers with one planted forbidden inference, answers rewritten to drop one requirement, and three samples of an
  SFT checkpoint (`answers.jsonl`), with Gemini's two calls per answer.
- `results/rl-dev/`, `results/rl-training/`, `results/rl-controls/`: RL checkpoint selection (core reward of two
  development answers per task, by GPT-6 Luna and by base Qwen3-8B) and per-rollout training logs (step, task,
  reward and its components, validity flags; completions removed) of the three RL runs compared in the appendix,
  plus test rows of the earlier validity-gated run.

## Not included

- Full texts of the source articles (open access, CC BY; `papers.jsonl` gives PMCID and DOI).
- The expert-alignment data: the 20 researcher-built maps of a separate planning study (participants' research
  plans), their expert ratings and the judge's calls on them.

## License

CC BY 4.0 (`LICENSE`). Each task is derived from an open-access article under a CC BY license; credit the source
article (`papers.jsonl`) as well as this dataset. The model outputs in `outputs/` and the judge and reward calls in
`results/` are released for reproducibility and are also subject to the terms of the models that produced them.
