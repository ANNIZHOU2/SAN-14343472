"""The pre-processing script standardised transcript paragraphs, removed empty and exact 
duplicate records, extracted adjacent paragraph context, and assigned firm, call-date and 
pre/post-GenAI identifiers. It then applied predefined AI-related keyword patterns to produce 
all_paragraphs.csv as the full cleaned corpus and ai_screening_results.csv as the 
keyword-screened candidate set for subsequent manual validation and analysis.

Expected input columns:
    company, call_date, quarter, speaker, speaker_type,
    paragraph_id, paragraph_text

Example:
    python code.py --input earnings_call_paragraphs.csv --output-dir results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# These lists are deliberately separate so the review file records why a
# paragraph was selected.
CORE_AI_TERMS = [
    r"\bartificial intelligence\b",
    r"\bAI\b",
    r"\bgenerative AI\b",
    r"\bGenAI\b",
    r"\bgenerative artificial intelligence\b",
    r"\bAI-powered\b",
    r"\bAI-enabled\b",
]

TECHNICAL_AI_TERMS = [
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bneural network\b",
    r"\blarge language model\b",
    r"\bLLM\b",
    r"\bgenerative model[s]?\b",
    r"\bfoundation model[s]?\b",
    r"\bretrieval[- ]augmented generation\b",
    r"\bRAG\b",
    r"\bfine[- ]tuning\b",
    r"\bprompt engineering\b",
    r"\bAI infrastructure\b",
    r"\bAI platform\b",
    r"\bAI assistant\b",
    r"\bAI agent[s]?\b",
    r"\bnatural language processing\b",
    r"\bcomputer vision\b",
    r"\btransformer model\b",
    r"\bdeep neural network\b",
]

# Representative LLM/generative-AI model names used during 2021-2024.
MODEL_TERMS = [
    r"\bChatGPT\b",
    r"\bGPT[- ]?3\.5\b",
    r"\bGPT[- ]?4\b",
    r"\bGPT[- ]?3\b",
    r"\bDALL[- ]?E\b",
    r"\bClaude\b",
    r"\bGemini\b",
    r"\bBard\b",
    r"\bLlama\b",
    r"\bPaLM\b",
    r"\bCopilot\b",
]


def compile_terms(terms: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(terms), flags=re.IGNORECASE)


CORE_PATTERN = compile_terms(CORE_AI_TERMS)
TECHNICAL_PATTERN = compile_terms(TECHNICAL_AI_TERMS)
MODEL_PATTERN = compile_terms(MODEL_TERMS)


def find_matches(text: str, pattern: re.Pattern[str]) -> list[str]:
    return sorted({match.group(0) for match in pattern.finditer(text)}, key=str.lower)


def classify_hits(text: str) -> dict[str, str | int]:
    core_hits = find_matches(text, CORE_PATTERN)
    technical_hits = find_matches(text, TECHNICAL_PATTERN)
    model_hits = find_matches(text, MODEL_PATTERN)

    hit_types: list[str] = []
    if core_hits:
        hit_types.append("core_ai")
    if technical_hits:
        hit_types.append("technical_ai")
    if model_hits:
        hit_types.append("model_name")

    return {
        "core_ai_hits": "; ".join(core_hits),
        "technical_ai_hits": "; ".join(technical_hits),
        "model_name_hits": "; ".join(model_hits),
        "hit_types": "; ".join(hit_types),
        "keyword_hit": int(bool(hit_types)),
    }


def read_text_file(path: Path) -> str:
    """Read common transcript encodings without changing the source file."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}")


def extract_date(text: str) -> pd.Timestamp | pd.NaT:
    """Extract common YYYY-MM-DD, YYYY_MM_DD, YYYYMMDD, or YYYY-MM dates."""
    patterns = [
        r"(?<!\d)(20\d{2})[-_./](0?[1-9]|1[0-2])[-_./](0?[1-9]|[12]\d|3[01])(?!\d)",
        r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0?[1-9]|[12]\d|3[01])(?!\d)",
        r"(?<!\d)(20\d{2})[-_./](0?[1-9]|1[0-2])(?!\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        parts = match.groups()
        if len(parts) == 3:
            year, month, day = parts
            value = f"{year}-{int(month):02d}-{int(day):02d}"
        else:
            year, month = parts
            value = f"{year}-{int(month):02d}-01"
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    return pd.NaT


def read_txt_directory(path: Path) -> pd.DataFrame:
    """Read every TXT file recursively; blank-line blocks are paragraphs."""
    rows: list[dict[str, object]] = []
    txt_files = sorted(path.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under: {path}")

    for txt_path in txt_files:
        raw_text = read_text_file(txt_path)
        blocks = [
            block.strip()
            for block in re.split(r"\r?\n\s*\r?\n+", raw_text)
            if block.strip()
        ]
        relative_path = txt_path.relative_to(path)
        file_date = extract_date(str(relative_path))
        # Use the first folder as the company when the directory is organized
        # as company/transcript.txt; otherwise use the file stem.
        company = relative_path.parts[0] if len(relative_path.parts) > 1 else txt_path.stem
        call_id = str(relative_path.with_suffix(""))

        for paragraph_id, block in enumerate(blocks, start=1):
            rows.append({
                "company": company,
                "call_date": file_date,
                "quarter": "",
                "speaker": "",
                "speaker_type": "",
                "paragraph_id": paragraph_id,
                "paragraph_text": block,
                "source_file": str(relative_path),
                "call_id": call_id,
            })

    return pd.DataFrame(rows)


def read_input(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return read_txt_directory(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    required = {"company", "call_date", "paragraph_text"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input is missing required columns: " + ", ".join(missing)
        )


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(df)
    df = df.copy()

    if "quarter" not in df.columns:
        df["quarter"] = ""
    if "speaker" not in df.columns:
        df["speaker"] = ""
    if "speaker_type" not in df.columns:
        df["speaker_type"] = ""
    if "source_file" not in df.columns:
        df["source_file"] = ""
    if "call_id" not in df.columns:
        df["call_id"] = ""
    if "paragraph_id" not in df.columns:
        df["paragraph_id"] = (
            df.groupby(["company", "call_date"], dropna=False).cumcount() + 1
        )

    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
    # Missing dates are retained for review instead of silently assigning a
    # date. They receive an "unknown" period and are excluded from pre/post
    # comparisons until the source filename/path is corrected.

    df["paragraph_text_original"] = df["paragraph_text"].fillna("").astype(str)
    df["paragraph_text_clean"] = (
        df["paragraph_text_original"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["paragraph_text_clean"].str.len() > 0].copy()

    # Remove exact duplicate records before keyword screening. The text is
    # normalized so whitespace and line-break differences do not create
    # separate records. Keep the first record for reproducibility.
    duplicate_keys = [
        "company",
        "call_date",
        "quarter",
        "paragraph_id",
        "paragraph_text_clean",
    ]
    duplicate_mask = df.duplicated(subset=duplicate_keys, keep="first")
    duplicate_rows = df[duplicate_mask].copy()
    df = df[~duplicate_mask].copy()

    # Keep chronology stable within each earnings call.
    df = df.sort_values(
        ["company", "call_date", "paragraph_id"], kind="stable"
    ).reset_index(drop=True)

    hit_data = df["paragraph_text_clean"].apply(classify_hits).apply(pd.Series)
    df = pd.concat([df, hit_data], axis=1)

    # Context is attached before filtering, so it always means the true
    # adjacent paragraph in the original call, not the adjacent hit.
    group = df.groupby(["company", "call_date"], dropna=False)
    df["prev_paragraph"] = group["paragraph_text_original"].shift(1)
    df["next_paragraph"] = group["paragraph_text_original"].shift(-1)

    cutoff = pd.Timestamp("2022-11-30")
    df["period"] = df["call_date"].map(
        lambda value: (
            "unknown" if pd.isna(value)
            else ("post" if value >= cutoff else "pre")
        )
    )
    df["year"] = df["call_date"].dt.year.astype("Int64")
    df["quarter_num"] = df["call_date"].dt.quarter.astype("Int64")
    df["year_quarter"] = df["call_date"].dt.to_period("Q").astype(str)

    return df, duplicate_rows


def export_outputs(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_df = df[df["keyword_hit"] == 1].copy()

    ai_df.to_csv(
        output_dir / "ai_screening_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df.to_csv(
        output_dir / "all_paragraphs.csv",
        index=False,
        encoding="utf-8-sig",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen earnings-call paragraphs for AI-related text."
    )
    parser.add_argument(
        "--input", default=None,
        help="Input CSV/XLSX file or directory containing TXT files.",
    )
    parser.add_argument(
        "--input-dir", default="transcripts",
        help="Directory recursively containing TXT files. Defaults to transcripts.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for screening outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    source = read_input(input_path)
    prepared, duplicate_rows = prepare_data(source)
    export_outputs(prepared, output_dir)

    print(f"Processed paragraphs after deduplication: {len(prepared):,}")
    print(f"Exact duplicate rows removed: {len(duplicate_rows):,}")
    print(f"Keyword-hit paragraphs: {int(prepared['keyword_hit'].sum()):,}")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
