---
name: arxiv-topic-review
description: Search arXiv by topic, optionally match venues to CCF grades, read sources, and generate per-paper summaries plus an index report
---

# arXiv Topic Review Skill

This skill runs an end-to-end pipeline:
- Search arXiv for a topic
- Optionally match venue names to CCF grades
- Download and read LaTeX sources
- Generate a per-paper `summary_*.md` plus an `index.md` overview

## When to Use

Use this skill when you want a repeatable topic review workflow that yields readable summaries and a ranked shortlist.

## Quick Start

1) (Optional) Rebuild the bundled CCF index (only if you want to refresh it):
```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/build_ccf_index.py \
  --pdf "[PATH_TO_CCF_PDF]" \
  --out [YOUR_SKILLS_DIR]/arxiv-topic-review/assets/ccf_2022.json
```

2) Run the pipeline:
```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/arxiv_pipeline.py \
  --query "T2I JailBreak" \
  --target-count 5 \
  --use-ccf
```

Outputs are written to `./sample/summary_*.md` and `./sample/index.md` in your current working directory.

The skill ships with a prebuilt `assets/ccf_2022.json`, so it does **not** read the PDF at runtime.

Documentation (default display: English):
- `USAGE_EN.md` (English, detailed)
- `USAGE_ZH.md` (中文说明)

Bundled reference PDF:
- `assets/ccf_2022.pdf` (reference only)

## Key Options

- `--use-ccf`: Enable CCF venue matching and grading
- `--ccf-json`: Path to the CCF JSON index (defaults to bundled `assets/ccf_2022.json`)
- `--ranking`: `relevance` or `ccf_first` (default auto: `ccf_first` if `--use-ccf`)
- `--summary-mode`: `auto`, `llm`, or `extractive` (default: `auto`)
- `--language`: `en` or `zh` (default: `en`)
- `--llm-provider`: `openai`, `anthropic`, `codex`, or `claude`
- `--llm-model`: Model name (required for LLM mode)
- `--report-dir`: Output directory (default: `./sample`)

## LLM Setup (Optional)

Use environment variables or CLI flags:

OpenAI / Codex:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)

Anthropic / Claude:
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL` (optional)

If LLM configuration is missing, the pipeline falls back to `extractive` summaries.

## Dependencies

- Required: `arxiv`
- Optional for rebuilding CCF index: `ghostscript` (`gs`) or `pdfplumber`
- Optional for LLM summaries: `openai` and/or `anthropic`

Install as needed:
```bash
python3 -m pip install arxiv pdfplumber openai anthropic
```
