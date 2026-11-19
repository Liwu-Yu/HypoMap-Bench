# HypoMap-Bench

**Can Language Models Understand the Process of Hypothesis Validation?**  
[Liwu Yu](mailto:francisyu24@ucla.edu), [Jiarui Cui](mailto:jiaruicui99@ucla.edu), [Youngseung Jeon](mailto:ysj@ucla.edu), and [Xiang ‘Anthony’ Chen](mailto:xac@ucla.edu) · University of California, Los Angeles · [Paper (PDF)](paper/HypoMap-Bench.pdf)

Given a hypothesis map and evidence from a published study, a model decides which claims need revision and which should stay as they are. HypoMap-Bench contains **3,178 distinct tasks from 722 papers** in materials science and biomedicine. On the 479-task test set, supervised fine-tuning raised Qwen3-8B’s answer validity from 67% to 86%, raised correctness from 0.72 to 0.81, and reduced unnecessary edits from 38% to 20%.

## Get started

From the repository root with Python 3.12, run:

```bash
python -m pip install -r requirements.txt
python inference.py --model qwen3-8b --data-dir data/prompts/test --limit 1 --dry-run
python analysis/paper_tables.py
```

The dry run inspects a test prompt. The last command reads cached scores and writes the paper’s main results table and family-level results to `analysis/out/`. Both run without a GPU or API key.

- [Dataset guide](data/README.md) — split counts, record fields, source-paper identifiers, and model outputs. For inference, use the prompt-only files in `data/prompts/`. The task files also contain reference answers.
- [Experiment guide](docs/experiments.md) — generate answers, score all five criteria, and run supervised or reinforcement learning.

Code is [MIT licensed](LICENSE). Benchmark data are licensed under [CC BY 4.0](data/LICENSE). Credit the source article when reusing a task.
