#!/usr/bin/env python3
"""Build a CCF venue index from the 2022 PDF list."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import subprocess
import shutil


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

PUBLISHERS = {
    "acm",
    "ieee",
    "springer",
    "elsevier",
    "wiley",
    "siam",
    "mit",
    "mit press",
    "ios",
    "world scientific",
    "oxford",
    "cambridge",
    "sage",
}


def normalize_name(text: str, aggressive: bool = False) -> str:
    s = text.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if aggressive:
        tokens = [t for t in s.split() if t and t not in STOPWORDS]
        s = " ".join(tokens)
    return s


def split_line(line: str) -> List[str]:
    parts = re.split(r"\s{2,}|\t|;|；|、", line)
    cleaned: List[str] = []
    for part in parts:
        seg = part.strip()
        if not seg:
            continue
        seg = re.sub(r"^[0-9]+[\).\s]+", "", seg)
        seg = re.sub(r"\s+", " ", seg).strip()
        if seg:
            cleaned.append(seg)
    return cleaned


def extract_abbr(seg: str) -> Optional[str]:
    for match in re.finditer(r"\(([^)]+)\)", seg):
        candidate = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z0-9&\- ]{2,24}", candidate):
            return candidate
    return None


def parse_entries(lines: Iterable[str]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    level: Optional[str] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.search(r"A\s*类", line):
            level = "A"
            continue
        if re.search(r"B\s*类", line):
            level = "B"
            continue
        if re.search(r"C\s*类", line):
            level = "C"
            continue
        if not level:
            continue

        parts = split_line(line)

        # Heuristic: rows with a leading index often have [index, abbr, full name, publisher, url]
        if len(parts) >= 3 and re.match(r"^\d+$", parts[0]):
            abbr = parts[1].strip()
            name_parts: List[str] = []
            for seg in parts[2:]:
                seg = seg.strip()
                if not seg:
                    continue
                lower = seg.lower()
                if lower.startswith("http"):
                    tail = seg.split(" ", 1)[1].strip() if " " in seg else ""
                    if tail:
                        name_parts.append(tail)
                    continue
                if lower in PUBLISHERS:
                    continue
                if seg == abbr:
                    continue
                name_parts.append(seg)
            name = " ".join(name_parts).strip()
            if abbr and name and re.search(r"[A-Za-z]", name):
                entries.append(_make_entry(name=name, abbr=abbr, level=level))
            continue

        for seg in parts:
            if not re.search(r"[A-Za-z]", seg):
                continue
            if seg.lower().startswith("http"):
                continue
            if seg.lower() in PUBLISHERS:
                continue
            abbr = extract_abbr(seg)
            name = seg
            if abbr:
                name = seg.replace(f"({abbr})", "").strip()
            name = re.sub(r"\s+", " ", name).strip()
            if not name:
                name = seg
            entries.append(_make_entry(name=name, abbr=abbr or "", level=level))

    return entries


def _make_entry(name: str, abbr: str, level: str) -> Dict[str, str]:
    entry = {
        "name": name,
        "abbr": abbr or "",
        "level": level,
        "norm": normalize_name(name, aggressive=False),
        "norm_aggr": normalize_name(name, aggressive=True),
    }
    if entry["abbr"]:
        entry["abbr_norm"] = normalize_name(entry["abbr"], aggressive=False)
        entry["abbr_norm_aggr"] = normalize_name(entry["abbr"], aggressive=True)
    return entry


def merge_wrapped_lines(lines: Iterable[str]) -> List[str]:
    merged: List[str] = []
    buffer = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d+\b", line):
            if buffer:
                merged.append(buffer)
            buffer = line
            continue
        if buffer:
            if re.search(r"[A-Za-z]", line) and not re.search(r"A\\s*类|B\\s*类|C\\s*类", line):
                buffer = f"{buffer} {line}"
                continue
            merged.append(buffer)
            buffer = ""
        merged.append(line)

    if buffer:
        merged.append(buffer)
    return merged


def dedupe_entries(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for entry in entries:
        key = (entry.get("norm"), entry.get("abbr_norm"), entry.get("level"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def extract_lines_from_pdf(pdf_path: Path) -> List[str]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return extract_lines_via_gs(pdf_path)

    lines: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
    return merge_wrapped_lines(lines)


def extract_lines_via_gs(pdf_path: Path) -> List[str]:
    if not shutil.which("gs"):
        raise SystemExit(
            "Error: ghostscript (gs) not found. Install it or use pdfplumber to parse the PDF."
        )
    result = subprocess.run(
        ["gs", "-sDEVICE=txtwrite", "-o", "-", str(pdf_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    text = result.stdout.decode("utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("GPL Ghostscript"):
            continue
        if line.startswith("Copyright"):
            continue
        if line.startswith("Processing pages"):
            continue
        if line.startswith("Loading font"):
            continue
        if line.startswith("Page "):
            continue
        lines.append(line)
    return merge_wrapped_lines(lines)


def build_index(pdf_path: Path) -> Dict[str, object]:
    lines = extract_lines_from_pdf(pdf_path)
    entries = parse_entries(lines)
    entries = dedupe_entries(entries)
    return {
        "source": str(pdf_path),
        "entry_count": len(entries),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CCF venue index from the PDF")
    parser.add_argument("--pdf", required=True, help="Path to the CCF PDF")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)

    if not pdf_path.exists():
        raise SystemExit(f"Error: PDF not found: {pdf_path}")

    index = build_index(pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {index['entry_count']} entries to {out_path}")


if __name__ == "__main__":
    main()
