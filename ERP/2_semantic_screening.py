"""A supplementary vectorisation procedure was used to identify paragraphs that may have 
contained AI-related discourse without predefined AI keywords. Paragraphs excluded from 
the manually verified AI corpus were embedded using BGE-M3 and compared with verified 
AI excerpts using cosine similarity. Results were ranked by similarity in the output file, with 
each excluded paragraph matched to its five closest AI reference excerpts. Approximately 
600 high-ranking matches with cosine similarity scores of at least 0.76 were manually 
reviewed. No additional substantive AI-related paragraphs were identified. Similarity scores 
were used only to prioritise manual review and did not automatically alter the formal AI 
corpus.

Required environment variables:
    OPENAI_API_KEY
    OPENAI_BASE_URL

Required package:
    python -m pip install openai numpy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from openai import OpenAI


MODEL_NAME = "bge-m3"
DEFAULT_BATCH_SIZE = 8
DEFAULT_TOP_K = 5
MAX_CHARS = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-file", type=Path, default=None)
    parser.add_argument("--raw-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def normalise_for_overlap(value: Any) -> str:
    """Compare full paragraphs after removing inconsequential whitespace/case."""
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown"))
    return text.strip("._")[:100] or "unknown"


def select_input_file(value: Path | None, output_dir: Path, filename: str) -> Path:
    return value if value is not None else output_dir / filename


def require_columns(headers: list[str], needed: set[str], source: Path) -> None:
    missing = sorted(needed - set(headers))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV using UTF-8 where possible, then Excel's common fallback."""
    decode_errors: list[UnicodeDecodeError] = []
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                rows = list(csv.DictReader(file))
            print(f"Read {path.name} using {encoding} encoding.")
            return rows
        except UnicodeDecodeError as exc:
            decode_errors.append(exc)
    raise decode_errors[-1]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        rows = read_csv_rows(path)
    else:
        raise ValueError(f"Only .csv inputs are supported: {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def record_id(row: dict[str, Any], row_number: int, prefix: str) -> str:
    call_id = clean_text(row.get("call_id"))
    paragraph_id = clean_text(row.get("paragraph_id"))
    if call_id and paragraph_id:
        return f"{prefix}:{call_id}:{paragraph_id}"
    return f"{prefix}:row:{row_number}"


def source_identity(row: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return a stable company/date/paragraph identity across source exports.

    Database-generated call_id values can differ between the all-paragraph
    export and the manually reviewed raw export. Company, call date, and
    paragraph number remain stable for the same transcript segment.
    """
    company = clean_text(row.get("company")).casefold()
    call_date = clean_text(row.get("call_date")).casefold()
    paragraph_id = clean_text(row.get("paragraph_id")).casefold()
    if company and call_date and paragraph_id:
        return company, call_date, paragraph_id
    return None


def load_input_data(all_file: Path, raw_file: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = read_rows(all_file)
    raw_rows = read_rows(raw_file)
    require_columns(list(all_rows[0]), {"paragraph_text_original"}, all_file)
    require_columns(list(raw_rows[0]), {"paragraph_text_original", "ai_excerpt"}, raw_file)
    return all_rows, raw_rows


def build_corpora(
    all_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Excel and CSV occasionally encode the same accented character differently.
    # The call/paragraph key is therefore the primary exclusion rule.
    raw_source_ids = {
        identifier
        for row in raw_rows
        if (identifier := source_identity(row)) is not None
    }
    # Exclude every full paragraph appearing in ai_screening_results.csv,
    # regardless of whether its ai_relevant value is yes or no. The raw
    # corpus is the manually reviewed universe and must not re-enter the
    # omitted-paragraph candidate pool.
    raw_originals = {
        normalise_for_overlap(row["paragraph_text_original"])
        for row in raw_rows
        if normalise_for_overlap(row["paragraph_text_original"])
    }

    excluded: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for source_row, row in enumerate(all_rows, start=2):
        original = clean_text(row["paragraph_text_original"])
        normalized_original = normalise_for_overlap(original)
        if (
            not original
            or source_identity(row) in raw_source_ids
            or normalized_original in raw_originals
        ):
            continue
        record = {
            "record_id": record_id(row, source_row, "excluded"),
            "source_row": source_row,
            "text": original,
            "text_hash": text_hash(original),
            "truncated": len(original) > MAX_CHARS,
            "eligible_for_similarity": True,
            "source": row,
        }
        excluded.append(record)
        candidates.append(record)

    references: list[dict[str, Any]] = []
    for source_row, row in enumerate(raw_rows, start=2):
        excerpt = clean_text(row["ai_excerpt"])
        if (
            not excerpt
            or clean_text(row.get("ai_relevant")).casefold() != "yes"
        ):
            continue
        references.append(
            {
                "record_id": record_id(row, source_row, "reference"),
                "source_row": source_row,
                "text": excerpt,
                "text_hash": text_hash(excerpt),
                "truncated": len(excerpt) > MAX_CHARS,
                "source": row,
            }
        )
    return excluded, candidates, references


def batched(values: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def truncate_for_embedding(record: dict[str, Any]) -> str:
    return record["text"][:MAX_CHARS]


def embed_missing(
    client: OpenAI,
    records: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, np.ndarray]:
    unique_by_hash = {record["text_hash"]: record for record in records}
    missing = list(unique_by_hash.values())
    vectors: dict[str, np.ndarray] = {}
    print(f"Unique texts to embed: {len(missing)}")

    for batch_number, batch in enumerate(batched(missing, batch_size), start=1):
        inputs = [truncate_for_embedding(record) for record in batch]
        for attempt in range(1, 6):
            try:
                response = client.embeddings.create(model=MODEL_NAME, input=inputs)
                batch_embeddings = [None] * len(inputs)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding
                if any(vector is None for vector in batch_embeddings):
                    raise RuntimeError("Gateway returned an incomplete embedding response.")
                batch_vectors = {}
                for record, vector in zip(batch, batch_embeddings, strict=True):
                    array = np.asarray(vector, dtype=np.float32)
                    norm = np.linalg.norm(array)
                    if norm == 0:
                        raise ValueError(f"Zero vector returned for {record['record_id']}")
                    # Normalisation makes the later dot product equal cosine similarity.
                    array /= norm
                    batch_vectors[record["text_hash"]] = array
                vectors.update(batch_vectors)
                break
            except Exception as exc:  # Network/rate-limit errors vary by Athena deployment.
                if attempt == 5:
                    raise RuntimeError(f"Embedding request failed after 5 attempts: {exc}") from exc
                wait_seconds = 2**attempt
                print(f"Batch {batch_number} failed ({exc}); retrying in {wait_seconds}s.")
                time.sleep(wait_seconds)
        if batch_number % 25 == 0 or batch_number == (len(missing) + batch_size - 1) // batch_size:
            print(f"Completed {min(batch_number * batch_size, len(missing))}/{len(missing)} unique texts")
    return vectors


def write_similarity_results(
    excluded: list[dict[str, Any]],
    references: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    output_file: Path,
    top_k: int,
) -> None:
    unique_references = {record["text_hash"]: record for record in references}
    reference_hashes = sorted(unique_references)
    reference_matrix = np.vstack([vectors[digest] for digest in reference_hashes])
    top_k = min(top_k, len(reference_hashes))

    fields = [
        "excluded_record_id", "excluded_source_row", "excluded_company", "excluded_call_id",
        "excluded_paragraph_id", "excluded_text", "rank", "cosine_similarity",
        "reference_record_id", "reference_source_row", "reference_company", "reference_call_id",
        "reference_paragraph_id", "reference_ai_excerpt",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for batch in batched(excluded, 256):
            matrix = np.vstack([vectors[record["text_hash"]] for record in batch])
            scores = matrix @ reference_matrix.T
            candidate_indices = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
            for row_index, record in enumerate(batch):
                ranked_indices = candidate_indices[row_index][
                    np.argsort(scores[row_index, candidate_indices[row_index]])[::-1]
                ]
                source = record["source"]
                for rank, ref_index in enumerate(ranked_indices, start=1):
                    reference = unique_references[reference_hashes[ref_index]]
                    ref_source = reference["source"]
                    writer.writerow(
                        {
                            "excluded_record_id": record["record_id"],
                            "excluded_source_row": record["source_row"],
                            "excluded_company": source.get("company"),
                            "excluded_call_id": source.get("call_id"),
                            "excluded_paragraph_id": source.get("paragraph_id"),
                            "excluded_text": record["text"],
                            "rank": rank,
                            "cosine_similarity": f"{scores[row_index, ref_index]:.6f}",
                            "reference_record_id": reference["record_id"],
                            "reference_source_row": reference["source_row"],
                            "reference_company": ref_source.get("company"),
                            "reference_call_id": ref_source.get("call_id"),
                            "reference_paragraph_id": ref_source.get("paragraph_id"),
                            "reference_ai_excerpt": reference["text"],
                        }
                    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("--batch-size and --top-k must both be positive.")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise EnvironmentError("OPENAI_API_KEY and OPENAI_BASE_URL must both be configured.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_file = select_input_file(args.all_file, args.output_dir, "all_paragraphs.csv")
    raw_file = select_input_file(args.raw_file, args.output_dir, "ai_screening_results.csv")
    all_rows, raw_rows = load_input_data(all_file, raw_file)
    excluded, candidates, references = build_corpora(all_rows, raw_rows)
    if not candidates or not references:
        raise RuntimeError("The excluded or AI-reference corpus is empty; no comparison can be run.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0)
    vectors = embed_missing(client, candidates + references, args.batch_size)
    write_similarity_results(
        candidates,
        references,
        vectors,
        args.output_dir / "excluded_top_matches.csv",
        args.top_k,
    )

    print(f"Excluded paragraphs: {len(excluded):,}")
    print(f"Excluded keyword-hit candidates compared: {len(candidates):,}")
    print(f"AI reference excerpts: {len(references):,}")
    print(f"Model: {MODEL_NAME}")
    print(f"Similarity results: {args.output_dir / 'excluded_top_matches.csv'}")


if __name__ == "__main__":
    main()
