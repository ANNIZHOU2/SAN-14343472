"""BERTopic was applied to the manually verified ai_excerpt texts to identify latent thematic 
structures, using sentence embeddings, n-gram features and a customised corporate stop-word 
list. The CORPORATE_STOP_WORDS list represents the final version developed after 
several rounds of revision and rerunning the model, and the resulting topic summary was 
exported to BERTopic_summary.xlsx."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


# The requested filename is BERTopic.py. Remove its directory before importing
# the package to avoid a case-insensitive Windows module-name collision.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path for path in sys.path
    if Path(path or ".").resolve() != SCRIPT_DIR
]

from bertopic import BERTopic  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


DEFAULT_INPUT = Path("results") / "ai_screening_results.csv"
DEFAULT_OUTPUT = Path("results") / "BERTopic_summary.xlsx"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


CORPORATE_STOP_WORDS = {
    "we", "our", "you", "they", "it", "this", "that", "the", "and",
    "of", "to", "in", "for", "on", "with", "is", "are", "was", "were",
    "be", "as", "at", "by", "from", "have", "has", "had", "will",
    "would", "can", "could", "should", "do", "does", "did", "think",
    "said", "say", "right", "just", "really", "also", "well", "now",
    "one", "going", "get", "got", "see", "look", "like", "about",
    "company", "business", "customer", "customers", "quarter", "year",
    "revenue", "growth", "million", "billion", "percent", "question",
    "call", "today", "thank", "thanks", "good", "new", "time", "things",
    "point", "first", "second", "part", "lot", "many", "make", "made",
    "some", "because", "sort", "them", "years", "doing", "unknown",
    "every", "early", "pro", "total", "impact", "gross", "margin",
    "approximately", "sequentially", "fiscal", "organic", "expect",
    "expected", "strong", "continued", "continue", "result", "results",
    "mentioned", "discussed", "including", "across", "around", "overall",
    "really", "very", "much", "more", "most", "many", "number",
}


# Product/model names are replaced before topic-word extraction so BERTopic
# does not create separate topics for ChatGPT, Gemini, Llama, and Copilot.
AI_MODEL_PATTERN = re.compile(
    r"\b(?:chatgpt|gpt[- ]?(?:3(?:\.5)?|4(?:o|\.1)?|5)|gemini|claude|"
    r"llama(?:\s*[23])?|copilot|dall[- ]?e|bard|palm|firefly|bedrock|"
    r"midjourney|mi(?:100|200|300|325|350|400))\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a topic-level BERTopic summary from ai_excerpt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV file containing the ai_excerpt column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Excel file containing topic-level results.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=30,
        help="Minimum number of documents per topic.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def normalize_model_names(text: str) -> str:
    """Collapse named AI models/products into one semantic token."""
    text = re.sub(r"\bgenerative\s+AI\b|\bgen\s*AI\b|\bGenAI\b", "generativeai", text, flags=re.IGNORECASE)
    text = AI_MODEL_PATTERN.sub("aimodel", text)
    text = re.sub(r"(?:\baimodel\b\s*){2,}", "aimodel ", text, flags=re.IGNORECASE)
    return text


def split_into_two_sentence_chunks(text: str) -> list[str]:
    """Split an excerpt into short, non-overlapping two-sentence chunks."""
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", text)
        if unit.strip()
    ]
    return [
        " ".join(units[index:index + 2])
        for index in range(0, len(units), 2)
    ]


def load_documents(input_path: Path) -> list[str]:
    dataframe = pd.read_csv(input_path)

    if "ai_excerpt" not in dataframe.columns:
        raise ValueError("The CSV file does not contain an 'ai_excerpt' column.")

    documents = []
    for value in dataframe["ai_excerpt"].tolist():
        text = normalize_model_names(clean_text(value))
        if text:
            documents.extend(split_into_two_sentence_chunks(text))

    if not documents:
        raise ValueError("No non-empty ai_excerpt values were found.")

    return documents


def serialize_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.min_topic_size < 2:
        raise ValueError("--min-topic-size must be at least 2.")

    documents = load_documents(input_path)
    print(f"Documents used: {len(documents):,}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    vectorizer_model = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | CORPORATE_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=8,
        max_df=0.85,
        token_pattern=r"(?u)\b(?:ai|aimodel|[a-zA-Z][a-zA-Z-]{2,})\b",
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        nr_topics=20,
        calculate_probabilities=False,
        verbose=True,
    )

    # BERTopic assigns topics internally, but document-level assignments are
    # deliberately not exported because this script produces only a summary.
    topic_model.fit_transform(documents)

    topic_info = topic_model.get_topic_info().copy()

    # Topic -1 is BERTopic's outlier group, not a substantive theme.
    if "Topic" in topic_info.columns:
        topic_info = topic_info[topic_info["Topic"] != -1].copy()

    rename_map = {
        "Topic": "topic",
        "Count": "count",
        "Name": "topic_name",
        "Representation": "top_keywords",
        "Representative_Docs": "representative_excerpts",
    }
    topic_summary = topic_info.rename(columns=rename_map)

    if "topic_name" in topic_summary.columns:
        topic_summary["topic_name"] = topic_summary["topic_name"].map(
            lambda value: (
                "AI model ecosystem"
                if "aimodel" in str(value).lower()
                else value
            )
        )

    for column in topic_summary.columns:
        topic_summary[column] = topic_summary[column].map(serialize_value)

    preferred_columns = [
        "topic",
        "count",
        "topic_name",
        "top_keywords",
        "representative_excerpts",
    ]
    existing_columns = [
        column for column in preferred_columns
        if column in topic_summary.columns
    ]
    topic_summary = topic_summary[existing_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    topic_summary.to_excel(output_path, index=False, sheet_name="topic_summary")

    print(f"Topics exported: {len(topic_summary):,}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
