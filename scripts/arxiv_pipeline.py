#!/usr/bin/env python3
"""End-to-end arXiv topic review pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import textwrap
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
ASSETS_DIR = SKILL_DIR / "assets"
DEFAULT_CCF_JSON = ASSETS_DIR / "ccf_2022.json"

STOPWORDS = {
    "international",
    "conference",
    "conf",
    "symposium",
    "workshop",
    "annual",
    "proceedings",
    "on",
    "of",
    "the",
    "and",
    "for",
    "journal",
    "transactions",
    "letters",
    "society",
    "association",
}


class CcfIndex:
    def __init__(self, entries: List[Dict[str, str]]):
        self.entries = entries
        self.exact: Dict[str, List[Dict[str, str]]] = {}
        self.exact_aggr: Dict[str, List[Dict[str, str]]] = {}

        for entry in entries:
            for key in [entry.get("norm"), entry.get("abbr_norm")]:
                if key:
                    self.exact.setdefault(key, []).append(entry)
            for key in [entry.get("norm_aggr"), entry.get("abbr_norm_aggr")]:
                if key:
                    self.exact_aggr.setdefault(key, []).append(entry)


def normalize_name(text: str, aggressive: bool = False) -> str:
    s = text.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if aggressive:
        tokens = [t for t in s.split() if t and t not in STOPWORDS]
        s = " ".join(tokens)
    return s


def load_ccf_index(path: Path) -> CcfIndex:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    for entry in entries:
        if "norm" not in entry:
            entry["norm"] = normalize_name(entry.get("name", ""), aggressive=False)
        if "norm_aggr" not in entry:
            entry["norm_aggr"] = normalize_name(entry.get("name", ""), aggressive=True)
        if entry.get("abbr") and "abbr_norm" not in entry:
            entry["abbr_norm"] = normalize_name(entry.get("abbr", ""), aggressive=False)
        if entry.get("abbr") and "abbr_norm_aggr" not in entry:
            entry["abbr_norm_aggr"] = normalize_name(entry.get("abbr", ""), aggressive=True)
    return CcfIndex(entries)


def level_rank(level: str, order: List[str]) -> int:
    try:
        return order.index(level)
    except ValueError:
        return len(order) + 1


def select_best(entries: List[Dict[str, str]], order: List[str]) -> Dict[str, str]:
    return sorted(entries, key=lambda e: level_rank(e.get("level", ""), order))[0]


def match_candidate(
    candidate: str, index: CcfIndex, order: List[str], threshold: float
) -> Optional[Tuple[Dict[str, str], float, str]]:
    norm = normalize_name(candidate, aggressive=False)
    norm_aggr = normalize_name(candidate, aggressive=True)

    if norm in index.exact:
        best = select_best(index.exact[norm], order)
        return best, 1.0, "exact"
    if norm_aggr in index.exact_aggr:
        best = select_best(index.exact_aggr[norm_aggr], order)
        return best, 0.98, "aggr"

    best_entry: Optional[Dict[str, str]] = None
    best_score = 0.0
    for entry in index.entries:
        for key in [entry.get("norm_aggr"), entry.get("abbr_norm_aggr")]:
            if not key:
                continue
            score = SequenceMatcher(None, norm_aggr, key).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry
    if best_entry and best_score >= threshold:
        return best_entry, best_score, "fuzzy"
    return None


def extract_candidate_venues(journal_ref: str, comment: str) -> List[str]:
    candidates: List[str] = []
    patterns = [
        r"accepted to (.+)",
        r"accepted at (.+)",
        r"to appear in (.+)",
        r"published in (.+)",
        r"presented at (.+)",
        r"in (.+)",
    ]

    for text in [journal_ref, comment]:
        if not text:
            continue
        candidates.append(text.strip())
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip())
        for match in re.finditer(r"\(([^)]+)\)", text):
            cand = match.group(1).strip()
            if cand:
                candidates.append(cand)

    expanded: List[str] = []
    for cand in candidates:
        expanded.append(cand)
        for part in re.split(r";|,", cand):
            part = part.strip()
            if part:
                expanded.append(part)

    seen = set()
    uniq: List[str] = []
    for cand in expanded:
        if cand in seen:
            continue
        seen.add(cand)
        uniq.append(cand)
    return uniq


def match_venue(
    journal_ref: str,
    comment: str,
    index: CcfIndex,
    order: List[str],
    threshold: float,
) -> Optional[Dict[str, str]]:
    candidates = extract_candidate_venues(journal_ref, comment)
    best: Optional[Tuple[Dict[str, str], float, str]] = None
    for cand in candidates:
        result = match_candidate(cand, index, order, threshold)
        if not result:
            continue
        entry, score, method = result
        if not best:
            best = (entry, score, method)
            continue
        if level_rank(entry.get("level", ""), order) < level_rank(best[0].get("level", ""), order):
            best = (entry, score, method)
        elif score > best[1]:
            best = (entry, score, method)

    if not best:
        return None

    entry, score, method = best
    matched = dict(entry)
    matched["match_score"] = f"{score:.3f}"
    matched["match_method"] = method
    return matched


def extract_arxiv_id(entry_id: str) -> str:
    match = re.search(r"arxiv\.org/(abs|pdf)/([^v/]+)(v\d+)?", entry_id)
    if match:
        return match.group(2)
    return entry_id.rsplit("/", 1)[-1]


def search_arxiv(query: str, max_results: int) -> List[object]:
    try:
        import arxiv
    except ImportError as exc:
        raise SystemExit(
            "Error: arxiv package not installed. Install with: python3 -m pip install arxiv"
        ) from exc

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    for member in tar.getmembers():
        member_path = path / member.name
        if not str(member_path.resolve()).startswith(str(path.resolve())):
            raise RuntimeError("Unsafe path in tar archive")
    tar.extractall(path)


def download_source(arxiv_id: str, cache_dir: Path) -> Optional[Path]:
    ensure_dir(cache_dir)
    tar_path = cache_dir / f"{arxiv_id}.tar.gz"
    if tar_path.exists():
        return tar_path

    url = f"https://arxiv.org/src/{arxiv_id}"
    req = Request(url, headers={"User-Agent": "arxiv-topic-review/0.1"})
    try:
        with urlopen(req, timeout=60) as response:
            with tar_path.open("wb") as fh:
                shutil.copyfileobj(response, fh)
    except Exception:
        if tar_path.exists():
            tar_path.unlink(missing_ok=True)
        return None
    return tar_path


def extract_source(tar_path: Path, cache_dir: Path, arxiv_id: str) -> Optional[Path]:
    extract_dir = cache_dir / arxiv_id
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return extract_dir

    try:
        ensure_dir(extract_dir)
        with tarfile.open(tar_path, "r:*") as tar:
            safe_extract(tar, extract_dir)
    except Exception:
        return None
    return extract_dir


def score_tex_file(path: Path) -> Tuple[int, int]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return (0, 0)
    score = 0
    if "\\documentclass" in content:
        score += 3
    if "\\begin{document}" in content:
        score += 3
    if path.stem.lower() in {"main", "paper", "article", "manuscript", "ms"}:
        score += 2
    length = len(content)
    return (score, length)


def find_main_tex(root: Path) -> Optional[Path]:
    tex_files = [
        p
        for p in root.rglob("*.tex")
        if "__MACOSX" not in p.parts and "auto" not in p.parts
    ]
    if not tex_files:
        return None
    if len(tex_files) == 1:
        return tex_files[0]
    scored = sorted(tex_files, key=lambda p: score_tex_file(p), reverse=True)
    return scored[0]


def strip_comments(line: str) -> str:
    out = []
    escaped = False
    for ch in line:
        if ch == "%" and not escaped:
            break
        if ch == "\\" and not escaped:
            escaped = True
            out.append(ch)
            continue
        escaped = False
        out.append(ch)
    return "".join(out)


def find_includes(text: str) -> List[str]:
    includes = []
    for match in re.finditer(r"\\(input|include)\{([^}]+)\}", text):
        includes.append(match.group(2).strip())
    return includes


def collect_tex(root: Path, entry: Path, visited: Optional[set] = None) -> str:
    if visited is None:
        visited = set()
    entry = entry.resolve()
    if entry in visited:
        return ""
    visited.add(entry)
    try:
        raw = entry.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    lines = [strip_comments(line) for line in raw.splitlines()]
    text = "\n".join(lines)

    content = [text]
    for inc in find_includes(text):
        inc_path = Path(inc)
        if not inc_path.suffix:
            inc_path = inc_path.with_suffix(".tex")
        if not inc_path.is_absolute():
            inc_path = (entry.parent / inc_path).resolve()
        if inc_path.exists():
            content.append(collect_tex(root, inc_path, visited))
    return "\n".join(content)


def extract_sections(tex: str) -> List[str]:
    sections = []
    for match in re.finditer(r"\\section\*?\{([^}]+)\}", tex):
        title = match.group(1).strip()
        if title:
            sections.append(title)
    return sections


def latex_to_text(tex: str) -> str:
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", tex)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^\}]*\})?", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_provider(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = name.lower().strip()
    if name in {"codex", "openai"}:
        return "openai"
    if name in {"claude", "anthropic"}:
        return "anthropic"
    return name


def resolve_model(provider: Optional[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if provider == "openai":
        return os.getenv("OPENAI_MODEL")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL")
    return None


def resolve_api_key(provider: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    return None


def resolve_base_url(provider: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_BASE_URL")
    return None


def llm_summarize(
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Install with: python3 -m pip install openai"
            ) from exc
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package not installed. Install with: python3 -m pip install anthropic"
            ) from exc
        client = Anthropic(api_key=api_key, base_url=base_url)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(part.text for part in message.content).strip()

    raise RuntimeError(f"Unsupported provider: {provider}")


def build_summary_prompt(paper: Dict[str, object], query: str, language: str, body: str) -> Tuple[str, str]:
    lang_label = "English" if language == "en" else "Chinese"
    system_prompt = (
        "You are a research assistant. Produce a concise paper summary for a topic review. "
        f"Write in {lang_label} and output Markdown with the headings specified."
    )

    user_prompt = textwrap.dedent(
        f"""
        Topic: {query}

        Paper metadata:
        Title: {paper.get('title')}
        Authors: {paper.get('authors')}
        Published: {paper.get('published')}
        arXiv ID: {paper.get('arxiv_id')}
        Journal ref: {paper.get('journal_ref')}
        Comment: {paper.get('comment')}

        Abstract:
        {paper.get('summary')}

        Section outline:
        {paper.get('section_outline', '')}

        Body snippet:
        {body}

        Write the summary with these headings (verbatim):
        ## Summary
        ## Key Contributions
        ## Methods
        ## Experiments
        ## Limitations
        ## Relevance to "{query}"
        """
    ).strip()

    return system_prompt, user_prompt


def summary_has_headings(text: str) -> bool:
    return bool(re.search(r"^##\\s+\\S+", text or "", re.MULTILINE))


def render_summary_markdown(
    paper: Dict[str, object],
    summary_body: str,
    section_outline: List[str],
    summary_mode: str,
    language: str,
) -> str:
    lines = []
    lines.append(f"# {paper['title']}")
    lines.append(f"- arXiv ID: {paper['arxiv_id']}")
    lines.append(f"- arXiv URL: {paper['arxiv_url']}")
    lines.append(f"- Published: {paper['published']}")
    lines.append(f"- Authors: {paper['authors']}")
    lines.append(f"- CCF Grade: {paper['ccf_grade']}")
    if paper.get("ccf_venue"):
        lines.append(f"- CCF Venue: {paper['ccf_venue']}")
    if paper.get("ccf_match"): 
        lines.append(f"- CCF Match: {paper['ccf_match']}")
    lines.append(f"- Summary Mode: {summary_mode}")
    lines.append(f"- Language: {language}")
    lines.append("")
    lines.append("## Abstract")
    lines.append(paper.get("summary", "").strip() or "(missing)")
    lines.append("")
    lines.append("## Section Outline")
    if section_outline:
        for section in section_outline[:20]:
            lines.append(f"- {section}")
    else:
        lines.append("- (not available)")
    lines.append("")
    if summary_has_headings(summary_body):
        lines.append(summary_body.strip() if summary_body else "(missing)")
        lines.append("")
    else:
        lines.append("## Summary")
        lines.append(summary_body.strip() if summary_body else "(missing)")
        lines.append("")
    if paper.get("source_note"):
        lines.append("## Source Note")
        lines.append(paper["source_note"])
        lines.append("")
    return "\n".join(lines)


def write_index(
    report_dir: Path,
    query: str,
    target_count: int,
    ranking: str,
    use_ccf: bool,
    papers: List[Dict[str, object]],
) -> None:
    lines = []
    lines.append("# arXiv Topic Review")
    lines.append(f"- Query: {query}")
    lines.append(f"- Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- Target Count: {target_count}")
    lines.append(f"- Ranking: {ranking}")
    lines.append(f"- CCF Enabled: {use_ccf}")
    lines.append("")
    lines.append("| # | Title | arXiv | CCF | Venue | Summary | Status |")
    lines.append("| - | - | - | - | - | - | - |")

    for idx, paper in enumerate(papers, start=1):
        title = paper["title"].replace("|", " ")
        arxiv_id = paper["arxiv_id"]
        ccf = paper.get("ccf_grade", "arxiv")
        venue = paper.get("ccf_venue", "-")
        summary_file = paper.get("summary_file", "-")
        status = paper.get("status", "ok")
        summary_link = f"[{Path(summary_file).name}]({Path(summary_file).name})"
        lines.append(
            f"| {idx} | {title} | {arxiv_id} | {ccf} | {venue} | {summary_link} | {status} |"
        )

    index_path = report_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")


def choose_summary_mode(args: argparse.Namespace) -> str:
    if args.summary_mode != "auto":
        return args.summary_mode
    provider = resolve_provider(args.llm_provider)
    model = resolve_model(provider, args.llm_model) if provider else None
    return "llm" if provider and model else "extractive"


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv topic review pipeline")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--use-ccf", action="store_true")
    parser.add_argument("--ccf-json", default=str(DEFAULT_CCF_JSON))
    parser.add_argument("--ccf-fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--ccf-levels", default="A,B,C")
    parser.add_argument("--ranking", choices=["relevance", "ccf_first", "hybrid"], default="auto")
    parser.add_argument("--summary-mode", choices=["auto", "llm", "extractive"], default="auto")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-max-tokens", type=int, default=700)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--report-dir", default=str(Path.cwd() / "sample"))
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/arxiv-topic-review"))
    parser.add_argument("--max-body-chars", type=int, default=12000)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    ccf_index: Optional[CcfIndex] = None
    ccf_levels = [level.strip() for level in args.ccf_levels.split(",") if level.strip()]

    if args.use_ccf:
        ccf_json = Path(args.ccf_json)
        if not ccf_json.exists():
            raise SystemExit(
                "Error: CCF JSON not found. Rebuild it with build_ccf_index.py and place it in assets."
            )
        ccf_index = load_ccf_index(ccf_json)

    ranking = args.ranking
    if ranking == "auto":
        ranking = "ccf_first" if args.use_ccf else "relevance"

    results = search_arxiv(args.query, args.max_candidates)
    papers: List[Dict[str, object]] = []

    for idx, result in enumerate(results):
        arxiv_id = extract_arxiv_id(getattr(result, "entry_id", ""))
        if hasattr(result, "get_short_id"):
            arxiv_id = re.sub(r"v\\d+$", "", result.get_short_id())
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        authors = ", ".join([a.name for a in result.authors]) if result.authors else ""
        published = result.published.strftime("%Y-%m-%d") if result.published else ""
        paper = {
            "idx": idx,
            "title": result.title.strip().replace("\n", " "),
            "summary": result.summary.strip().replace("\n", " "),
            "authors": authors,
            "published": published,
            "entry_id": getattr(result, "entry_id", ""),
            "arxiv_id": arxiv_id,
            "arxiv_url": arxiv_url,
            "journal_ref": getattr(result, "journal_ref", "") or "",
            "comment": getattr(result, "comment", "") or "",
        }

        if args.use_ccf and ccf_index:
            match = match_venue(
                paper["journal_ref"],
                paper["comment"],
                ccf_index,
                ccf_levels,
                args.ccf_fuzzy_threshold,
            )
            if match:
                paper["ccf_grade"] = match.get("level", "arxiv")
                paper["ccf_venue"] = match.get("name", "")
                paper["ccf_match"] = f"{match.get('match_method')}:{match.get('match_score')}"
            else:
                paper["ccf_grade"] = "arxiv"
        else:
            paper["ccf_grade"] = "arxiv"

        papers.append(paper)

    if ranking == "relevance":
        ranked = papers
    elif ranking == "ccf_first":
        ranked = sorted(
            papers,
            key=lambda p: (level_rank(p.get("ccf_grade", "arxiv"), ccf_levels), p["idx"]),
        )
    else:
        ranked = sorted(
            papers,
            key=lambda p: (level_rank(p.get("ccf_grade", "arxiv"), ccf_levels), -p["idx"]),
        )

    selected = ranked[: args.target_count]

    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)
    cache_dir = Path(args.cache_dir)

    summary_mode = choose_summary_mode(args)

    for paper in selected:
        slug = re.sub(r"[^a-z0-9]+", "_", paper["title"].lower()).strip("_")
        if not slug:
            slug = paper["arxiv_id"].replace("/", "_")
        slug = slug[:60]
        summary_path = report_dir / f"summary_{slug}.md"
        if summary_path.exists():
            summary_path = report_dir / f"summary_{slug}_{paper['arxiv_id'].replace('/', '_')}.md"
        if summary_path.exists() and args.skip_existing:
            paper["summary_file"] = str(summary_path)
            paper["status"] = "skipped"
            continue

        source_note = ""
        sections: List[str] = []
        body_text = ""

        tar_path = download_source(paper["arxiv_id"], cache_dir)
        if tar_path:
            extract_dir = extract_source(tar_path, cache_dir, paper["arxiv_id"])
            if extract_dir:
                main_tex = find_main_tex(extract_dir)
                if main_tex:
                    tex = collect_tex(extract_dir, main_tex)
                    sections = extract_sections(tex)
                    paper["section_outline"] = ", ".join(sections[:20])
                    plain = latex_to_text(tex)
                    body_text = plain[: args.max_body_chars]
                else:
                    source_note = "No .tex entrypoint found in source archive."
            else:
                source_note = "Failed to extract source archive."
        else:
            source_note = "Failed to download source archive."

        paper["source_note"] = source_note

        summary_body = ""
        used_mode = summary_mode

        if summary_mode == "llm":
            provider = resolve_provider(args.llm_provider)
            model = resolve_model(provider, args.llm_model) if provider else None
            if not provider or not model:
                used_mode = "extractive"
            else:
                api_key = resolve_api_key(provider, args.llm_api_key)
                if not api_key:
                    used_mode = "extractive"
                else:
                    base_url = resolve_base_url(provider, args.llm_base_url)
                    system_prompt, user_prompt = build_summary_prompt(
                        paper, args.query, args.language, body_text
                    )
                    try:
                        summary_body = llm_summarize(
                            provider,
                            model,
                            api_key,
                            base_url,
                            system_prompt,
                            user_prompt,
                            args.llm_max_tokens,
                            args.llm_temperature,
                        )
                    except Exception as exc:
                        used_mode = "extractive"
                        summary_body = f"LLM summary failed: {exc}"

        if used_mode == "extractive":
            summary_body = (
                paper.get("summary", "").strip()
                or "Extractive summary only. Abstract unavailable."
            )

        paper["summary_file"] = str(summary_path)
        paper["status"] = "ok" if not source_note else "partial"

        summary_markdown = render_summary_markdown(
            paper,
            summary_body,
            sections,
            used_mode,
            args.language,
        )
        summary_path.write_text(summary_markdown, encoding="utf-8")

    write_index(report_dir, args.query, args.target_count, ranking, args.use_ccf, selected)


if __name__ == "__main__":
    main()
