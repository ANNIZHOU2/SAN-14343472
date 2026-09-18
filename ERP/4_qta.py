"""This appendix describes the code used to aggregate the manually coded AI excerpts in 
ai_paragraphs_semantically_encoded.csv into call-level and paragraph-level datasets and to 
examine changes in corporate AI discourse across periods and firm types. The code produces 
descriptive statistics, regression results, robustness checks and supporting figures and tables 
for the empirical analysis.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


THEMES = [
    "AI strategy and investment",
    "AI infrastructure and computing",
    "Generative AI and LLM",
    "AI productivity and enterprise applications",
    "AI content and customer experience",
    "AI security, risk and responsibility",
]
ATTITUDES = ["positive", "neutral", "negative", "mixed", "not_applicable"]
FULL_PERIOD_PARAGRAPHS = 41454


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=r"ai_paragraphs_semantically_encoded.csv",
        help="Path to the semantically encoded CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"qta_results",
        help="Output directory. Defaults to qta_results in the repository root.",
    )
    parser.add_argument(
        "--full-input",
        default=r"results\all_paragraphs.csv",
        help="Full-period paragraph table used for corpus and call denominators.",
    )
    parser.add_argument(
        "--metadata-input",
        default=r"earnings_call_metadata.csv",
        help="Metadata file containing the company sample_group labels.",
    )
    return parser.parse_args()


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pp(x: float) -> str:
    return f"{100 * x:+.1f} percentage points"


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}")


def make_call_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["company"].astype(str)
        + "|"
        + df["call_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df.get("call_id", df["call_date"].astype(str)).astype(str)
    )


def load_calls(path: Path, full_path: Path, metadata_path: Path) -> pd.DataFrame:
    df = read_csv_with_fallback(path)
    full = read_csv_with_fallback(full_path)
    metadata = read_csv_with_fallback(metadata_path)
    required = {"company", "call_date", "ai_excerpt", "attitude", "theme"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
    df = df.loc[df["call_date"].notna()].copy()
    df["ai_nonempty"] = df["ai_excerpt"].fillna("").astype(str).str.strip().ne("")
    full_required = {"company", "call_date", "call_id", "paragraph_id"}
    full_missing = full_required - set(full.columns)
    if full_missing:
        raise ValueError(f"Missing required columns in full-period file: {sorted(full_missing)}")
    full["call_date"] = pd.to_datetime(full["call_date"], errors="coerce")
    full = full.loc[full["call_date"].notna()].copy()
    if {"ticker", "sample_group"}.issubset(metadata.columns):
        group_map = metadata.drop_duplicates("ticker").set_index("ticker")["sample_group"]
        full["sample_group"] = full["company"].map(group_map).fillna("unclassified")
    else:
        full["sample_group"] = "unclassified"
    df["call_key"] = make_call_key(df)
    full["call_key"] = make_call_key(full)
    full["year_quarter"] = full["call_date"].dt.to_period("Q").astype(str)
    df["year_quarter"] = df["call_date"].dt.to_period("Q").astype(str)
    df["post"] = (df["call_date"] >= pd.Timestamp("2023-01-01")).astype(int)
    df["transition_q4_2022"] = (
        (df["call_date"] >= pd.Timestamp("2022-10-01"))
        & (df["call_date"] < pd.Timestamp("2023-01-01"))
    )

    calls = (
        full.groupby(["call_key", "company", "sample_group", "call_date", "year_quarter"], as_index=False)
        .agg(total_paragraphs=("paragraph_id", "size"))
    )
    calls["post"] = (calls["call_date"] >= pd.Timestamp("2023-01-01")).astype(int)
    coded_counts = df.groupby("call_key", as_index=False).agg(ai_paragraphs=("ai_nonempty", "sum"))
    calls = calls.merge(coded_counts, on="call_key", how="left")
    calls["ai_paragraphs"] = calls["ai_paragraphs"].fillna(0).astype(int)
    calls["ai_presence"] = (calls["ai_paragraphs"] > 0).astype(int)
    calls["ai_share"] = calls["ai_paragraphs"] / calls["total_paragraphs"].clip(lower=1)
    calls["log_total_paragraphs"] = np.log1p(calls["total_paragraphs"])

    # Paragraph-level labels are collapsed to call-level shares.
    labeled = df.loc[df["ai_nonempty"]].copy()
    for attitude in ATTITUDES:
        labeled[f"att_{attitude}"] = (labeled["attitude"] == attitude).astype(int)
    for theme in THEMES:
        labeled[f"theme_{theme}"] = labeled["theme"].fillna("").map(
            lambda x: int(theme in str(x).split("; "))
        )
    label_cols = [f"att_{x}" for x in ATTITUDES] + [f"theme_{x}" for x in THEMES]
    label_means = labeled.groupby("call_key")[label_cols].mean().reset_index()
    calls = calls.merge(label_means, on="call_key", how="left")
    calls[label_cols] = calls[label_cols].fillna(0)
    calls["risk_responsibility_share"] = calls[
        ["theme_AI security, risk and responsibility"]
    ].sum(axis=1)
    calls["positive_share"] = calls["att_positive"]
    calls["not_applicable_share"] = calls["att_not_applicable"]
    calls["distinct_theme_count"] = calls[[f"theme_{x}" for x in THEMES]].gt(0).sum(axis=1)
    calls.attrs["screened_rows"] = len(df)
    calls.attrs["coded_rows"] = int(df["ai_nonempty"].sum())
    calls.attrs["full_period_paragraphs"] = len(full)
    return calls


def summarize_periods(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "ai_presence": "AI call presence",
        "ai_share": "AI paragraph share of full call text",
        "ai_paragraphs": "Coded AI paragraphs per call",
        "positive_share": "Positive attitude share",
        "risk_responsibility_share": "Security/risk/responsibility share",
        "distinct_theme_count": "Distinct themes per call",
    }
    for period, subset in [("Pre-GenAI (through 2022Q3)", calls[calls["call_date"] < "2022-10-01"]),
                           ("Transition (2022Q4)", calls[calls["transition_q4_2022"] if "transition_q4_2022" in calls else calls["call_date"].between("2022-10-01", "2022-12-31")]),
                           ("Post-GenAI (from 2023Q1)", calls[calls["call_date"] >= "2023-01-01"])]:
        for key, label in metrics.items():
            rows.append({"period": period, "metric": label, "mean": subset[key].mean(), "n_calls": len(subset)})
    return pd.DataFrame(rows)


def theme_attitude_table(calls: pd.DataFrame, prefix: str, labels: list[str]) -> pd.DataFrame:
    rows = []
    pre = calls[calls["call_date"] < "2022-10-01"]
    post = calls[calls["call_date"] >= "2023-01-01"]
    for label in labels:
        key = f"{prefix}{label}"
        pre_mean, post_mean = pre[key].mean(), post[key].mean()
        rows.append({
            "label": label,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "change": post_mean - pre_mean,
        })
    return pd.DataFrame(rows)


def make_longitudinal_table(calls: pd.DataFrame) -> pd.DataFrame:
    value_cols = {
        "ai_presence": "ai_call_presence",
        "ai_share": "ai_paragraph_share",
        "positive_share": "positive_share",
        "risk_responsibility_share": "risk_responsibility_share",
    }
    value_cols.update({f"theme_{theme}": f"theme_{theme}" for theme in THEMES})
    table = calls.groupby("year_quarter").agg(
        n_calls=("call_key", "nunique"),
        **{out: (col, "mean") for col, out in value_cols.items()},
    ).reset_index()
    return table


def make_heterogeneity_table(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = calls[calls["call_date"] < "2022-10-01"]
    post = calls[calls["call_date"] >= "2023-01-01"]
    for group in sorted(calls["sample_group"].dropna().unique()):
        for theme in THEMES:
            key = f"theme_{theme}"
            pre_group = pre[pre["sample_group"] == group]
            post_group = post[post["sample_group"] == group]
            pre_mean = pre_group[key].mean()
            post_mean = post_group[key].mean()
            rows.append({
                "sample_group": group,
                "theme": theme,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "change": post_mean - pre_mean,
                "pre_calls": len(pre_group),
                "post_calls": len(post_group),
            })
    return pd.DataFrame(rows)


def make_attitude_detail(path: Path, metadata_path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    coded = read_csv_with_fallback(path)
    coded = coded.loc[coded["ai_excerpt"].fillna("").astype(str).str.strip().ne("")].copy()
    coded["call_date"] = pd.to_datetime(coded["call_date"], errors="coerce")
    coded = coded.loc[coded["call_date"].notna()].copy()
    coded["period"] = np.select(
        [coded["call_date"] < "2022-10-01", coded["call_date"] < "2023-01-01"],
        ["pre", "transition"],
        default="post",
    )
    metadata = read_csv_with_fallback(metadata_path)
    if {"ticker", "sample_group"}.issubset(metadata.columns):
        group_map = metadata.drop_duplicates("ticker").set_index("ticker")["sample_group"]
        coded["sample_group"] = coded["company"].map(group_map).fillna("unclassified")
    else:
        coded["sample_group"] = "unclassified"

    counts = coded.groupby(["period", "attitude"], as_index=False).size().rename(columns={"size": "n_paragraphs"})
    totals = counts.groupby("period")["n_paragraphs"].transform("sum")
    counts["share"] = counts["n_paragraphs"] / totals

    group_counts = coded.groupby(["sample_group", "period", "attitude"], as_index=False).size().rename(columns={"size": "n_paragraphs"})
    group_totals = group_counts.groupby(["sample_group", "period"])["n_paragraphs"].transform("sum")
    group_counts["share"] = group_counts["n_paragraphs"] / group_totals

    mixed = coded[coded["attitude"] == "mixed"].copy()
    opportunity = r"opportun|excited|transform|productiv|growth|enhanc|accelerat|value|differentiat|promise|optimis|bullish|benefit"
    governance = r"privacy|security|risk|responsib|ethic|trust|transparen|accountab|copyright|indemnif|provenance|safety|hallucinat|misuse|bias"
    mixed_features = {
        "mixed_total": len(mixed),
        "mixed_opportunity_or_benefit": int(mixed["ai_excerpt"].str.contains(opportunity, case=False, regex=True, na=False).sum()),
        "mixed_governance_or_risk": int(mixed["ai_excerpt"].str.contains(governance, case=False, regex=True, na=False).sum()),
        "mixed_both_opportunity_and_governance": int((mixed["ai_excerpt"].str.contains(opportunity, case=False, regex=True, na=False) & mixed["ai_excerpt"].str.contains(governance, case=False, regex=True, na=False)).sum()),
    }
    mixed[["company", "call_date", "sample_group", "ai_excerpt"]].head(20).to_csv(
        out_dir / "mixed_attitude_examples.csv", index=False
    )
    return counts, group_counts, mixed_features


def _clustered_ols(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, coefficient_positions: list[int]) -> list[tuple[float, float, float, float]]:
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros_like(bread)
    for cluster in np.unique(clusters):
        idx = clusters == cluster
        xu = X[idx].T @ residual[idx]
        meat += np.outer(xu, xu)
    n, k, n_clusters = len(y), X.shape[1], len(np.unique(clusters))
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / max(n - k, 1))
    variance = correction * bread @ meat @ bread
    output = []
    for position in coefficient_positions:
        standard_error = math.sqrt(max(variance[position, position], 0))
        z_value = beta[position] / standard_error if standard_error else np.nan
        output.append((beta[position], standard_error, z_value, normal_pvalue(z_value)))
    return output


def make_paragraph_attitude_regressions(path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coded = read_csv_with_fallback(path)
    coded = coded.loc[coded["ai_excerpt"].fillna("").astype(str).str.strip().ne("")].copy()
    coded["call_date"] = pd.to_datetime(coded["call_date"], errors="coerce")
    coded = coded.loc[coded["call_date"].notna()].copy()
    coded["period"] = np.select(
        [coded["call_date"] < "2022-10-01", coded["call_date"] < "2023-01-01"],
        ["pre", "transition"],
        default="post",
    )
    coded = coded.loc[coded["period"].isin(["pre", "post"])].copy()
    coded["post"] = (coded["period"] == "post").astype(float)
    coded["year_quarter"] = coded["call_date"].dt.to_period("Q").astype(str)
    coded["call_key"] = make_call_key(coded)
    metadata = read_csv_with_fallback(metadata_path)
    group_map = metadata.drop_duplicates("ticker").set_index("ticker")["sample_group"]
    coded["sample_group"] = coded["company"].map(group_map).fillna("unclassified")

    company_fe = pd.get_dummies(coded["company"], drop_first=True, dtype=float).to_numpy()
    quarter_fe = pd.get_dummies(coded["year_quarter"], drop_first=True, dtype=float).to_numpy()
    clusters = coded["call_key"].astype(str).to_numpy()
    standard_X = np.column_stack([np.ones(len(coded)), coded[["post"]].to_numpy(), company_fe, quarter_fe])
    interaction_groups = [
        "enterprise_software",
        "semis_ai_infra",
        "ai_forward_saas",
        "data_cloud_security",
    ]
    interaction_X = np.column_stack([
        np.ones(len(coded)),
        *[(coded["post"] * (coded["sample_group"] == group).astype(float)).to_numpy() for group in interaction_groups],
        company_fe,
        quarter_fe,
    ])

    standard_rows, interaction_rows = [], []
    for attitude in ["positive", "neutral", "negative", "mixed"]:
        y = (coded["attitude"] == attitude).astype(float).to_numpy()
        coef, se, z_value, p_value = _clustered_ols(standard_X, y, clusters, [1])[0]
        standard_rows.append({
            "level": "paragraph",
            "outcome": attitude,
            "coefficient": coef,
            "cluster_se": se,
            "z_value": z_value,
            "p_value": p_value,
            "n_paragraphs": len(coded),
            "n_calls": coded["call_key"].nunique(),
            "fixed_effects": "company + year_quarter",
            "cluster": "call_key",
        })
        estimates = _clustered_ols(interaction_X, y, clusters, list(range(1, 1 + len(interaction_groups))))
        for group, (group_coef, group_se, group_z, group_p) in zip(interaction_groups, estimates):
            interaction_rows.append({
                "level": "paragraph",
                "outcome": attitude,
                "comparison_group": group,
                "base_group": "platform_bigtech",
                "interaction_coefficient": group_coef,
                "cluster_se": group_se,
                "z_value": group_z,
                "p_value": group_p,
                "n_paragraphs": len(coded),
                "n_calls": coded["call_key"].nunique(),
                "fixed_effects": "company + year_quarter",
                "cluster": "call_key",
            })
    return pd.DataFrame(standard_rows), pd.DataFrame(interaction_rows)


def make_call_attitude_interactions(calls: pd.DataFrame) -> pd.DataFrame:
    data = calls.loc[
        (calls["call_date"] < "2022-10-01")
        | (calls["call_date"] >= "2023-01-01")
    ].copy()
    # The call-level analysis is defined over calls containing at least one
    # semantically coded AI paragraph; paragraph-level estimation below uses
    # the paragraph observations and clusters inference by the same calls.
    data = data.loc[data["ai_paragraphs"] > 0].copy()
    data["post"] = (data["call_date"] >= "2023-01-01").astype(float)
    interaction_groups = [
        "enterprise_software",
        "semis_ai_infra",
        "ai_forward_saas",
        "data_cloud_security",
    ]
    interaction_X = np.column_stack([
        np.ones(len(data)),
        data[["log_total_paragraphs"]].to_numpy(),
        *[(data["post"] * (data["sample_group"] == group).astype(float)).to_numpy() for group in interaction_groups],
        pd.get_dummies(data["company"], drop_first=True, dtype=float).to_numpy(),
        pd.get_dummies(data["year_quarter"], drop_first=True, dtype=float).to_numpy(),
    ])
    clusters = data["company"].astype(str).to_numpy()
    rows = []
    for attitude in ["positive", "neutral", "negative", "mixed"]:
        y = data[f"att_{attitude}"].astype(float).to_numpy()
        estimates = _clustered_ols(interaction_X, y, clusters, list(range(2, 2 + len(interaction_groups))))
        for group, (group_coef, group_se, group_z, group_p) in zip(interaction_groups, estimates):
            rows.append({
                "level": "call",
                "outcome": attitude,
                "comparison_group": group,
                "base_group": "platform_bigtech",
                "interaction_coefficient": group_coef,
                "cluster_se": group_se,
                "z_value": group_z,
                "p_value": group_p,
                "n_calls": len(data),
                "n_firms": data["company"].nunique(),
                "fixed_effects": "company + year_quarter",
                "cluster": "company",
            })
    return pd.DataFrame(rows)


def make_theme_regressions(path: Path, metadata_path: Path, calls: pd.DataFrame) -> pd.DataFrame:
    """Estimate theme changes at paragraph and call levels.

    The post indicator is identified with company fixed effects and
    quarter-of-year fixed effects. Full year-quarter fixed effects would be
    collinear with the post indicator once the transition quarter is removed.
    """
    coded = read_csv_with_fallback(path)
    coded = coded.loc[coded["ai_excerpt"].fillna("").astype(str).str.strip().ne("")].copy()
    coded["call_date"] = pd.to_datetime(coded["call_date"], errors="coerce")
    coded = coded.loc[coded["call_date"].notna()].copy()
    coded = coded.loc[
        (coded["call_date"] < "2022-10-01")
        | (coded["call_date"] >= "2023-01-01")
    ].copy()
    coded["post"] = (coded["call_date"] >= "2023-01-01").astype(float)
    coded["quarter_of_year"] = coded["call_date"].dt.quarter.astype(str)
    coded["call_key"] = make_call_key(coded)

    paragraph_company_fe = pd.get_dummies(coded["company"], drop_first=True, dtype=float).to_numpy()
    paragraph_season_fe = pd.get_dummies(coded["quarter_of_year"], drop_first=True, dtype=float).to_numpy()
    paragraph_X = np.column_stack([
        np.ones(len(coded)),
        coded[["post"]].to_numpy(),
        paragraph_company_fe,
        paragraph_season_fe,
    ])
    paragraph_clusters = coded["call_key"].astype(str).to_numpy()

    data = calls.loc[
        (calls["call_date"] < "2022-10-01")
        | (calls["call_date"] >= "2023-01-01")
    ].copy()
    data = data.loc[data["ai_paragraphs"] > 0].copy()
    data["quarter_of_year"] = data["call_date"].dt.quarter.astype(str)
    data["post"] = (data["call_date"] >= "2023-01-01").astype(float)
    call_company_fe = pd.get_dummies(data["company"], drop_first=True, dtype=float).to_numpy()
    call_season_fe = pd.get_dummies(data["quarter_of_year"], drop_first=True, dtype=float).to_numpy()
    call_X = np.column_stack([
        np.ones(len(data)),
        data[["post", "log_total_paragraphs"]].to_numpy(),
        call_company_fe,
        call_season_fe,
    ])
    call_clusters = data["company"].astype(str).to_numpy()

    rows = []
    for theme in THEMES:
        paragraph_y = coded["theme"].fillna("").map(
            lambda value: int(theme in str(value).split("; "))
        ).to_numpy(dtype=float)
        p_coef, p_se, p_z, p_p = _clustered_ols(
            paragraph_X, paragraph_y, paragraph_clusters, [1]
        )[0]
        rows.append({
            "level": "paragraph",
            "outcome": theme,
            "coefficient": p_coef,
            "cluster_se": p_se,
            "z_value": p_z,
            "p_value": p_p,
            "n_observations": len(coded),
            "n_clusters": coded["call_key"].nunique(),
            "fixed_effects": "company + quarter_of_year",
            "controls": "none",
            "cluster": "earnings_call",
        })

        c_coef, c_se, c_z, c_p = _clustered_ols(
            call_X, data[f"theme_{theme}"].to_numpy(dtype=float), call_clusters, [1]
        )[0]
        rows.append({
            "level": "call",
            "outcome": theme,
            "coefficient": c_coef,
            "cluster_se": c_se,
            "z_value": c_z,
            "p_value": c_p,
            "n_observations": len(data),
            "n_clusters": data["company"].nunique(),
            "fixed_effects": "company + quarter_of_year",
            "controls": "log_total_paragraphs",
            "cluster": "company",
        })
    return pd.DataFrame(rows)


def make_theme_interaction_regressions(path: Path, metadata_path: Path, calls: pd.DataFrame) -> pd.DataFrame:
    """Estimate post-by-company-type interactions for each theme."""
    interaction_groups = [
        "enterprise_software",
        "semis_ai_infra",
        "ai_forward_saas",
        "data_cloud_security",
    ]
    coded = read_csv_with_fallback(path)
    coded = coded.loc[coded["ai_excerpt"].fillna("").astype(str).str.strip().ne("")].copy()
    coded["call_date"] = pd.to_datetime(coded["call_date"], errors="coerce")
    coded = coded.loc[coded["call_date"].notna()].copy()
    coded = coded.loc[
        (coded["call_date"] < "2022-10-01")
        | (coded["call_date"] >= "2023-01-01")
    ].copy()
    coded["post"] = (coded["call_date"] >= "2023-01-01").astype(float)
    coded["quarter_of_year"] = coded["call_date"].dt.quarter.astype(str)
    coded["call_key"] = make_call_key(coded)
    group_map = read_csv_with_fallback(metadata_path).drop_duplicates("ticker").set_index("ticker")["sample_group"]
    coded["sample_group"] = coded["company"].map(group_map).fillna("unclassified")
    paragraph_company_fe = pd.get_dummies(coded["company"], drop_first=True, dtype=float).to_numpy()
    paragraph_season_fe = pd.get_dummies(coded["quarter_of_year"], drop_first=True, dtype=float).to_numpy()
    paragraph_X = np.column_stack([
        np.ones(len(coded)),
        *[(coded["post"] * (coded["sample_group"] == group).astype(float)).to_numpy() for group in interaction_groups],
        paragraph_company_fe,
        paragraph_season_fe,
    ])
    paragraph_clusters = coded["call_key"].astype(str).to_numpy()

    data = calls.loc[
        ((calls["call_date"] < "2022-10-01") | (calls["call_date"] >= "2023-01-01"))
        & (calls["ai_paragraphs"] > 0)
    ].copy()
    data["post"] = (data["call_date"] >= "2023-01-01").astype(float)
    data["quarter_of_year"] = data["call_date"].dt.quarter.astype(str)
    call_company_fe = pd.get_dummies(data["company"], drop_first=True, dtype=float).to_numpy()
    call_season_fe = pd.get_dummies(data["quarter_of_year"], drop_first=True, dtype=float).to_numpy()
    call_X = np.column_stack([
        np.ones(len(data)),
        data[["log_total_paragraphs"]].to_numpy(),
        *[(data["post"] * (data["sample_group"] == group).astype(float)).to_numpy() for group in interaction_groups],
        call_company_fe,
        call_season_fe,
    ])
    call_clusters = data["company"].astype(str).to_numpy()

    rows = []
    for theme in THEMES:
        paragraph_y = coded["theme"].fillna("").map(
            lambda value: int(theme in str(value).split("; "))
        ).to_numpy(dtype=float)
        p_estimates = _clustered_ols(
            paragraph_X, paragraph_y, paragraph_clusters, list(range(1, 1 + len(interaction_groups)))
        )
        call_estimates = _clustered_ols(
            call_X, data[f"theme_{theme}"].to_numpy(dtype=float), call_clusters, list(range(2, 2 + len(interaction_groups)))
        )
        for level, estimates, n_obs, n_clusters in [
            ("paragraph", p_estimates, len(coded), coded["call_key"].nunique()),
            ("call", call_estimates, len(data), data["company"].nunique()),
        ]:
            for group, (coef, se, z_value, p_value) in zip(interaction_groups, estimates):
                rows.append({
                    "level": level,
                    "outcome": theme,
                    "comparison_group": group,
                    "base_group": "platform_bigtech",
                    "coefficient": coef,
                    "cluster_se": se,
                    "z_value": z_value,
                    "p_value": p_value,
                    "n_observations": n_obs,
                    "n_clusters": n_clusters,
                    "fixed_effects": "company + quarter_of_year",
                    "controls": "log_total_paragraphs" if level == "call" else "none",
                    "cluster": "company" if level == "call" else "earnings_call",
                })
    return pd.DataFrame(rows)


def make_company_theme_table(heterogeneity: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Create the publication-ready 5x7 company-type/theme table.

    The six theme cells report pre/post changes in percentage points. The
    source heterogeneity table stores changes as proportions, so values are
    multiplied by 100 here and rounded only for presentation/export.
    """
    group_order = [
        "platform_bigtech",
        "enterprise_software",
        "semis_ai_infra",
        "ai_forward_saas",
        "data_cloud_security",
    ]
    group_labels = {
        "platform_bigtech": "Platform / Big Tech",
        "enterprise_software": "Enterprise software",
        "semis_ai_infra": "Semiconductors / AI infrastructure",
        "ai_forward_saas": "AI-forward SaaS",
        "data_cloud_security": "Data / cloud / cybersecurity",
    }
    theme_order = [
        "AI strategy and investment",
        "AI infrastructure and computing",
        "Generative AI and LLM",
        "AI productivity and enterprise applications",
        "AI content and customer experience",
        "AI security, risk and responsibility",
    ]
    theme_labels = {
        "AI strategy and investment": "AI strategy & investment",
        "AI infrastructure and computing": "AI infrastructure & computing",
        "Generative AI and LLM": "Generative AI & LLM",
        "AI productivity and enterprise applications": "AI productivity & enterprise applications",
        "AI content and customer experience": "AI content & customer experience",
        "AI security, risk and responsibility": "AI security, risk & responsibility",
    }

    table = (
        heterogeneity[heterogeneity["sample_group"].isin(group_order)]
        .pivot(index="sample_group", columns="theme", values="change")
        .reindex(index=group_order, columns=theme_order)
        .mul(100)
        .rename(index=group_labels, columns=theme_labels)
        .round(1)
        .reset_index()
        .rename(columns={"sample_group": "Company type"})
    )
    table.to_csv(out_dir / "table_4b_theme_changes_by_firm_type.csv", index=False)

    markdown = table.copy()
    for column in markdown.columns[1:]:
        markdown[column] = markdown[column].map(lambda value: f"{value:+.1f}")
    header = "| " + " | ".join(markdown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(markdown.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in markdown.itertuples(index=False, name=None)
    ]
    (out_dir / "table_4b_theme_changes_by_firm_type.md").write_text(
        "\n".join([header, separator] + body) + "\n", encoding="utf-8"
    )
    return table


def normal_pvalue(t: float) -> float:
    return math.erfc(abs(t) / math.sqrt(2))


def fit_fixed_effects(calls: pd.DataFrame, outcome: str, exclude_transition: bool = True) -> dict:
    data = calls.copy()
    if exclude_transition:
        data = data.loc[~data["call_date"].between("2022-10-01", "2022-12-31")].copy()
    data = data.dropna(subset=[outcome, "post", "company", "year_quarter"])
    if data["post"].nunique() < 2:
        return {"outcome": outcome, "coef": np.nan, "se": np.nan, "p_value": np.nan, "n": len(data)}

    base = data[["post", "log_total_paragraphs"]].astype(float)
    d_company = pd.get_dummies(data["company"], drop_first=True, dtype=float)
    d_period = pd.get_dummies(data["year_quarter"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(data)), base.to_numpy(), d_company.to_numpy(), d_period.to_numpy()])
    y = data[outcome].astype(float).to_numpy()
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    residual = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros_like(bread)
    groups = data["company"].astype(str).to_numpy()
    for group in np.unique(groups):
        idx = groups == group
        xg = X[idx]
        ug = residual[idx]
        xu = xg.T @ ug
        meat += np.outer(xu, xu)
    n, k, g = len(data), X.shape[1], len(np.unique(groups))
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    vcov = correction * bread @ meat @ bread
    se = math.sqrt(max(vcov[1, 1], 0))
    t_value = beta[1] / se if se else np.nan
    return {"outcome": outcome, "coef": beta[1], "se": se, "p_value": normal_pvalue(t_value) if se else np.nan, "n": n}


def make_company_type_attitude_change_figure(attitude_group: pd.DataFrame, out_dir: Path, plt) -> str:
    figure_dir = out_dir
    attitudes = ["positive", "neutral", "negative", "mixed"]
    group_order = [
        "ai_forward_saas",
        "data_cloud_security",
        "enterprise_software",
        "platform_bigtech",
        "semis_ai_infra",
    ]
    group_labels = {
        "ai_forward_saas": "AI SaaS firms",
        "data_cloud_security": "Data, cloud, and\nsecurity firms",
        "enterprise_software": "Enterprise software firms",
        "platform_bigtech": "Platform and big\ntech firms",
        "semis_ai_infra": "Semiconductor and AI\ninfrastructure firms",
    }
    base = attitude_group.pivot_table(
        index="sample_group",
        columns=["period", "attitude"],
        values="share",
        fill_value=0,
    ).reindex(group_order, fill_value=0)
    changes = pd.DataFrame(index=base.index)
    for attitude in attitudes:
        changes[attitude] = base.get(("post", attitude), 0) - base.get(("pre", attitude), 0)
    changes = changes.sort_values("positive", ascending=True)

    colors = {
        "positive": "#2E7D32",
        "neutral": "#7B8794",
        "negative": "#C62828",
        "mixed": "#EF8A17",
    }
    fig_height = max(6, 0.75 * len(changes))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = list(range(len(changes)))
    bar_height = 0.18
    offsets = {attitude: offset * bar_height for attitude, offset in zip(attitudes, [-1.5, -0.5, 0.5, 1.5])}
    for attitude in attitudes:
        ax.barh(
            [value + offsets[attitude] for value in y],
            changes[attitude].to_numpy() * 100,
            height=bar_height,
            color=colors[attitude],
            alpha=0.9,
            label=attitude.capitalize(),
        )
    ax.set_yticks(y)
    ax.set_yticklabels([group_labels.get(group, group) for group in changes.index], fontsize=10)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel(
        "Change in attitude share: post-GenAI minus pre-GenAI (percentage points)",
        fontsize=12,
    )
    fig.suptitle(
        "Company-type changes in AI attitudes after GenAI diffusion",
        fontsize=12,
        x=0.58,
        y=0.98,
        ha="center",
        va="top",
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.4, 1.01),
        fontsize=11,
    )
    fig.subplots_adjust(left=0.34, right=0.98, top=0.88, bottom=0.13)
    output_path = figure_dir / "figure_4_company_type_attitude_change.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def make_figures(calls: pd.DataFrame, out_dir: Path, attitude_group: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    # Figure 3 uses the same font family as Figures 1–2, but is intentionally
    # two points larger for readability with six theme labels.
    title_size = 14
    label_size = 12
    tick_size = 12
    legend_size = 12
    quarterly = calls.groupby("year_quarter").agg(
        ai_presence=("ai_presence", "mean"),
        ai_share=("ai_share", "mean"),
        positive_share=("positive_share", "mean"),
        **{f"att_{attitude}": (f"att_{attitude}", "mean") for attitude in ["positive", "neutral", "negative", "mixed"]},
        risk_share=("risk_responsibility_share", "mean"),
        **{f"theme_{theme}": (f"theme_{theme}", "mean") for theme in THEMES},
    ).reset_index()
    quarterly.to_csv(out_dir / "table_10_quarterly_trends.csv", index=False)
    quarterly["quarter"] = pd.PeriodIndex(quarterly["year_quarter"], freq="Q").to_timestamp()
    transition_start = pd.Timestamp("2022-10-01")
    post_cutoff = pd.Timestamp("2023-01-01")

    def add_period_markers(ax, cutoff_label: str = "_nolegend_"):
        ax.axvspan(
            transition_start,
            post_cutoff,
            color="lightgray",
            alpha=0.35,
            zorder=0,
            label="_nolegend_",
        )
        ax.axvline(
            post_cutoff,
            color="black",
            linestyle="--",
            linewidth=1,
            label=cutoff_label,
        )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(quarterly["quarter"], quarterly["ai_share"] * 100, marker="o", label="AI paragraph share")
    add_period_markers(ax, "GenAI diffusion cutoff")
    ax.set_ylabel("Percent of call paragraphs")
    ax.set_title("AI discussion intensity in earnings calls")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_5_ai_intensity_robustness.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    for theme in THEMES:
        ax.plot(
            quarterly["quarter"],
            quarterly[f"theme_{theme}"] * 100,
            marker="o",
            linewidth=2,
            markersize=4,
            label=theme,
        )
    add_period_markers(ax)
    ax.set_ylabel("Average call-level theme share (%)", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if getattr(ax, "legend_", None) is not None:
        ax.legend_.remove()
    fig.suptitle("Longitudinal change in all AI themes", y=0.98, fontsize=title_size)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        title="Theme",
        fontsize=legend_size,
        title_fontsize=legend_size,
        columnspacing=2.0,
        handlelength=2.5,
    )
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.08, right=0.98)
    fig.savefig(out_dir / "figure_1_longitudinal_themes.png", dpi=220)
    plt.close(fig)

    # Horizontal grouped bars showing the absolute pre/post change for each
    # theme and company type, matching the percentage-point values reported
    # in the results narrative.
    group_order = [
        "platform_bigtech",
        "enterprise_software",
        "ai_forward_saas",
        "data_cloud_security",
        "semis_ai_infra",
    ]
    group_labels = {
        "platform_bigtech": "Platform and\nbig tech",
        "enterprise_software": "Enterprise\nsoftware",
        "ai_forward_saas": "AI focused\nSaaS",
        "data_cloud_security": "Data, cloud, and\nsecurity",
        "semis_ai_infra": "Semiconductors and\nAI infrastructure",
    }
    theme_labels = {
        "AI strategy and investment": "AI strategy and investment",
        "AI infrastructure and computing": "AI infrastructure and computing",
        "Generative AI and LLM": "Generative AI and LLM",
        "AI productivity and enterprise applications": "AI productivity and enterprise applications",
        "AI content and customer experience": "AI content and customer experience",
        "AI security, risk and responsibility": "AI security, risk and responsibility",
    }
    pre_calls = calls[calls["call_date"] < "2022-10-01"]
    post_calls = calls[calls["call_date"] >= "2023-01-01"]
    changes = []
    for group in group_order:
        pre_group = pre_calls[pre_calls["sample_group"] == group]
        post_group = post_calls[post_calls["sample_group"] == group]
        changes.append([
            100 * (post_group[f"theme_{theme}"].mean() - pre_group[f"theme_{theme}"].mean())
            for theme in THEMES
        ])
    changes = np.asarray(changes)

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(group_order))
    bar_height = 0.12
    offsets = (np.arange(len(THEMES)) - (len(THEMES) - 1) / 2) * bar_height
    colors = plt.get_cmap("tab10")(np.arange(len(THEMES)))
    for idx, theme in enumerate(THEMES):
        ax.barh(
            y + offsets[idx],
            changes[:, idx],
            height=bar_height * 0.9,
            color=colors[idx],
            label=theme_labels[theme],
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([group_labels[group] for group in group_order], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Pre–post change in theme share (percentage points)", fontsize=12)
    fig.suptitle(
        "Company-type differences in thematic change after GenAI diffusion",
        fontsize=14,
        y=0.99,
        ha="center",
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    fig.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        fontsize=10,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.78, bottom=0.18)
    fig.savefig(out_dir / "figure_2_company_type_theme_change.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for attitude in ["positive", "neutral", "negative", "mixed"]:
        ax.plot(quarterly["quarter"], quarterly[f"att_{attitude}"] * 100, marker="o", label=f"{attitude.capitalize()} attitude")
    add_period_markers(ax)
    ax.set_ylabel("Share of coded AI paragraphs (%)")
    ax.set_title("Longitudinal change in AI attitudes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_3_longitudinal_attitudes.png", dpi=220)
    plt.close(fig)

    figure_4 = make_company_type_attitude_change_figure(attitude_group, out_dir, plt)
    return [
        "figure_1_longitudinal_themes.png",
        "figure_2_company_type_theme_change.png",
        "figure_3_longitudinal_attitudes.png",
        figure_4,
        "figure_5_ai_intensity_robustness.png",
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    full_input_path = Path(args.full_input)
    metadata_path = Path(args.metadata_input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calls = load_calls(input_path, full_input_path, metadata_path)
    summary = summarize_periods(calls)
    themes = theme_attitude_table(calls, "theme_", THEMES)
    longitudinal = make_longitudinal_table(calls)
    heterogeneity = make_heterogeneity_table(calls)
    attitude_detail, attitude_group, mixed_features = make_attitude_detail(input_path, metadata_path, out_dir)
    paragraph_attitude_regressions, paragraph_attitude_interactions = make_paragraph_attitude_regressions(input_path, metadata_path)
    call_attitude_interactions = make_call_attitude_interactions(calls)
    theme_regressions = make_theme_regressions(input_path, metadata_path, calls)
    theme_interaction_regressions = make_theme_interaction_regressions(input_path, metadata_path, calls)
    regressions = pd.DataFrame([
        fit_fixed_effects(calls, "ai_presence"),
        fit_fixed_effects(calls, "ai_share"),
        fit_fixed_effects(calls, "positive_share"),
        fit_fixed_effects(calls, "risk_responsibility_share"),
        *[fit_fixed_effects(calls, f"att_{attitude}") for attitude in ["positive", "neutral", "negative", "mixed"]],
        fit_fixed_effects(calls, "theme_Generative AI and LLM"),
        fit_fixed_effects(calls, "theme_AI productivity and enterprise applications"),
    ])

    paragraph_theme_regressions = theme_regressions[theme_regressions["level"] == "paragraph"].copy()
    call_theme_regressions = theme_regressions[theme_regressions["level"] == "call"].copy()
    paragraph_attitude_interactions = paragraph_attitude_interactions.copy()
    call_attitude_interactions = call_attitude_interactions.copy()
    attitude_firm_interactions = pd.concat(
        [paragraph_attitude_interactions, call_attitude_interactions],
        ignore_index=True,
        sort=False,
    )
    call_attitude_regressions = regressions[
        regressions["outcome"].isin(["positive", "neutral", "negative", "mixed"])
    ].copy()

    calls.to_csv(out_dir / "call_level_panel.csv", index=False)
    summary.to_csv(out_dir / "table_0_period_summary.csv", index=False)

    # 1. Theme descriptive statistics
    themes.to_csv(out_dir / "table_1_theme_descriptive_statistics.csv", index=False)

    # 2-4. Theme regressions and firm-type interactions
    paragraph_theme_regressions.to_csv(out_dir / "table_2_paragraph_theme_regressions.csv", index=False)
    call_theme_regressions.to_csv(out_dir / "table_3_call_theme_regressions.csv", index=False)
    theme_interaction_regressions.to_csv(out_dir / "table_4_theme_firm_type_interactions.csv", index=False)
    make_company_theme_table(heterogeneity, out_dir)

    # 5-8. Attitude descriptive statistics, regressions and interactions
    attitude_detail.to_csv(out_dir / "table_5_attitude_descriptive_statistics.csv", index=False)
    attitude_group.to_csv(out_dir / "table_5b_attitude_by_firm_type.csv", index=False)
    paragraph_attitude_regressions.to_csv(out_dir / "table_6_paragraph_attitude_regressions.csv", index=False)
    call_attitude_regressions.to_csv(out_dir / "table_7_call_attitude_regressions.csv", index=False)
    attitude_firm_interactions.to_csv(out_dir / "table_8_attitude_firm_type_interactions.csv", index=False)

    # 9. Robustness and supporting checks
    regressions.to_csv(out_dir / "table_9_robustness_checks.csv", index=False)

    figures = make_figures(calls, out_dir, attitude_group)

    print(f"Input: {input_path}")
    print(f"Full corpus: {full_input_path} ({calls.attrs.get('full_period_paragraphs', 'unknown'):,} paragraphs)")
    print(f"Metadata: {metadata_path}")
    print(f"Calls: {len(calls):,}; firms: {calls['company'].nunique():,}")
    print(f"Outputs: {out_dir}")
    print(f"Company-theme table: {out_dir / 'table_4b_theme_changes_by_firm_type.csv'}")
    print(f"Figures: {', '.join(figures) if figures else 'skipped (matplotlib unavailable)'}")


if __name__ == "__main__":
    main()
