# arXiv 主题综述 Skill（中文）

## 目标

该 skill 提供端到端流程：
- 主题检索 arXiv
- 可选的 CCF 期刊/会议分级匹配
- 下载并读取 LaTeX 源码
- 生成每篇论文的摘要文件与总览索引

适合一次性输入主题，自动产出结构化阅读报告。

## 已打包内容

- `assets/ccf_2022.json`：已构建好的 CCF 索引（运行时使用）
- `assets/ccf_2022.pdf`：CCF 2022 原始 PDF（仅供参考，不在运行时读取）
- `scripts/arxiv_pipeline.py`：主流程脚本
- `scripts/build_ccf_index.py`：可选重建 CCF JSON 的脚本

运行时不再读取 PDF，只读取 JSON。

## 快速开始

在任意工作目录运行：

```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/arxiv_pipeline.py \
  --query "T2I JailBreak" \
  --target-count 5 \
  --use-ccf
```

输出目录默认为当前工作目录下的 `./sample/`。

## 关键参数

- `--query`：检索主题（必填）
- `--target-count`：达到此数量停止
- `--max-candidates`：检索候选数量（默认 30）
- `--ranking`：`relevance`、`ccf_first` 或 `hybrid`
- `--use-ccf`：开启 CCF 分级
- `--ccf-json`：CCF JSON 路径（默认 `assets/ccf_2022.json`）
- `--ccf-levels`：等级顺序，例如 `A,B,C`
- `--ccf-fuzzy-threshold`：模糊匹配阈值（默认 0.88）
- `--summary-mode`：`auto`、`llm` 或 `extractive`
- `--language`：`en` 或 `zh`
- `--report-dir`：输出目录（默认 `./sample`）
- `--cache-dir`：源码缓存目录
- `--skip-existing`：跳过已存在的摘要文件

## 排序逻辑

- 未指定 `--ranking` 时的默认逻辑：
  - 开启 CCF 时默认 `ccf_first`
  - 未开启 CCF 时默认 `relevance`
- `ccf_first` 会优先按 `A > B > C > arxiv` 排序，再按相关性顺序

## CCF 匹配说明

- arXiv 的很多预印本没有正式期刊/会议信息，可能会保持 `arxiv`。
- 主要从 `journal_ref` 和 `comment` 字段提取 venue 信息。
- 既支持会议简称也支持全称。

## 摘要模式

- `extractive`：不依赖模型，仅使用摘要
- `llm`：使用模型生成结构化摘要
- `auto`：若配置了模型则用 LLM，否则回退到 `extractive`

## 支持的 LLM 提供方

- `openai`（别名 `codex`）
- `anthropic`（别名 `claude`）

环境变量：
- `OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选）
- `ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`（可选）
- `OPENAI_MODEL`、`ANTHROPIC_MODEL` 可用于默认模型

示例：

```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/arxiv_pipeline.py \
  --query "T2I JailBreak" \
  --target-count 5 \
  --use-ccf \
  --summary-mode llm \
  --llm-provider openai \
  --llm-model gpt-4o-mini
```

## 输出结构

每篇 `summary_*.md` 包含：
- 元信息（arXiv ID、日期、作者、CCF 等级）
- 摘要
- 章节大纲（若源码可用）
- 结构化总结内容

`index.md` 总览包含：
- 排名
- 标题
- arXiv ID
- CCF 等级
- 会议/期刊名
- 摘要文件链接
- 状态

## 可选：重建 CCF JSON

如需刷新 CCF 索引：

```bash
python3 [YOUR_SKILLS_DIR]/arxiv-topic-review/scripts/build_ccf_index.py \
  --pdf [PATH_TO_CCF_PDF] \
  --out [YOUR_SKILLS_DIR]/arxiv-topic-review/assets/ccf_2022.json
```

该脚本默认尝试使用 Ghostscript（`gs`）。如安装了 `pdfplumber` 也可使用。

## 常见问题

- 找不到 CCF JSON：
  - 确认 `assets/ccf_2022.json` 已存在
- LLM 失败：
  - 检查 API Key 与模型名称
- 无法读取源码：
  - 部分 arXiv 论文没有源代码包

