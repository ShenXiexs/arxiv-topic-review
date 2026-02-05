# arXiv Topic Review Skill (English)

## Purpose

This skill provides an end-to-end pipeline to:
- Search arXiv for a topic
- Optionally match venues to CCF grades
- Download and read LaTeX sources
- Produce per-paper summaries and an index report

It is designed to run as a single CLI command after installing the skill into your Codex skills directory.

## What Is Bundled

- `assets/ccf_2022.json`: Prebuilt CCF venue index (used at runtime when `--use-ccf` is enabled)
- `assets/ccf_2022.pdf`: The original CCF 2022 PDF for reference only
- `scripts/arxiv_pipeline.py`: Main pipeline CLI
- `scripts/build_ccf_index.py`: Rebuilds the CCF JSON from the PDF (optional)

The pipeline does not read the PDF at runtime. It only reads the JSON.

## Quick Start

Run the pipeline from any working directory:

```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/arxiv_pipeline.py \
  --query "T2I JailBreak" \
  --target-count 5 \
  --use-ccf
```

Outputs are written to `./knowledge/summary_*.md` and `./knowledge/index.md` in your current working directory.

## Core Arguments

- `--query`: Search query (required)
- `--target-count`: Number of papers to summarize before stopping
- `--max-candidates`: Number of arXiv candidates to fetch (default 30)
- `--ranking`: `relevance`, `ccf_first`, or `hybrid`
- `--use-ccf`: Enable CCF venue matching and grading
- `--ccf-json`: Path to CCF JSON (default `assets/ccf_2022.json`)
- `--ccf-levels`: Grade order, e.g. `A,B,C`
- `--ccf-fuzzy-threshold`: Fuzzy match threshold (default 0.88)
- `--summary-mode`: `auto`, `llm`, or `extractive`
- `--language`: `en` or `zh`
- `--report-dir`: Output directory (default `./knowledge`)
- `--cache-dir`: Cache directory for arXiv source downloads
- `--skip-existing`: Skip summary files that already exist

## Ranking Behavior

- If `--ranking` is not specified, the default is:
  - `ccf_first` when `--use-ccf` is enabled
  - `relevance` otherwise
- `ccf_first` ranks by grade order (`A` > `B` > `C` > `arxiv`) and then by relevance order

## CCF Matching Notes

- arXiv preprints often lack venue information, so many items will stay `arxiv`.
- Matching uses venue names found in `journal_ref` and `comment` fields.
- The index includes both abbreviations and full names.

## Summary Modes

- `extractive`: Uses the arXiv abstract only (no LLM required)
- `llm`: Uses an LLM to produce a structured summary
- `auto`: Uses LLM only if provider and model are configured

## LLM Providers

Supported providers:
- `openai` (alias `codex`)
- `anthropic` (alias `claude`)

Environment variables:
- `OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional)
- `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (optional)
- `OPENAI_MODEL` and `ANTHROPIC_MODEL` can provide defaults

Example:

```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/arxiv_pipeline.py \
  --query "T2I JailBreak" \
  --target-count 5 \
  --use-ccf \
  --summary-mode llm \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

If LLM configuration is missing, the pipeline falls back to `extractive`.

## Output Structure

Each `summary_*.md` includes:
- Metadata (arXiv ID, date, authors, CCF grade)
- Abstract
- Section outline (when LaTeX sources are available)
- Summary content

The `index.md` includes a table with:
- Rank
- Title
- arXiv ID
- CCF grade
- Venue
- Summary file link
- Status

## Rebuilding the CCF JSON (Optional)

If you want to refresh the CCF index from the PDF:

```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/build_ccf_index.py \
  --pdf [PATH_TO_CCF_PDF] \
  --out [YOUR_SKILLS_DIR]/arxiv-topic-review/assets/ccf_2022.json
```

This script tries to use Ghostscript (`gs`) first. If `pdfplumber` is installed, it can also be used.

## Troubleshooting

- Missing CCF JSON:
  - Ensure `assets/ccf_2022.json` is present in the skill directory
- LLM errors:
  - Verify API keys and model names
- Missing LaTeX sources:
  - Some arXiv entries do not provide full source archives

