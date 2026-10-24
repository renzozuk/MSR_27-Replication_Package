#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import pandas as pd

"""
Repository-Stratified Pull Request Sampling

This script generates the manual validation dataset used in the empirical study.
Starting from the keyword agreement dataset, it removes pull requests that are
unsuitable for manual qualitative analysis and produces a repository-stratified
random sample.

Filtering rules
---------------
A pull request is excluded when its description:

- is empty or contains only null-like values;
- contains only an issue-closing reference (e.g., "Fixes #123");
- contains fewer than 10 words.

Sampling strategy
-----------------
After filtering, the remaining pull requests are sampled using repository-based
stratified random sampling. Each repository contributes at least five pull
requests (when available), and the remaining sample is proportionally allocated
according to the number of valid pull requests in each repository.

Outputs
-------
The following files are generated inside the output directory:

- sampled_prs.xlsx
    Final stratified sample for manual validation.

- filtered_out_prs.xlsx
    Pull requests excluded during preprocessing together with the exclusion
    reason.

- sampling_report.xlsx
    Per-repository sampling statistics.

Usage
-----
python sample_prs.py \
    --input ../dataset/keywords/keywords_strategies_nfr_agreements.csv \
    --sample-size 1046

Arguments
---------
--input
    Input CSV or Excel dataset containing the pull requests.

--sample-size
    Number of pull requests to sample.
    Default: 1046.

--output-dir
    Directory where the generated files will be stored.
    Default: ../dataset/sampling

Configuration Constants
-----------------------
DEFAULT_SAMPLE_SIZE
    Default number of pull requests to sample.

MIN_WORDS
    Minimum number of words required in the PR description.

MIN_PER_REPO
    Minimum number of sampled pull requests per repository whenever possible.

RANDOM_SEED
    Random seed used to ensure reproducible sampling.
"""

DEFAULT_SAMPLE_SIZE = 1046
MIN_WORDS = 10
MIN_PER_REPO = 5
RANDOM_SEED = 42

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "sample"
)
INVALID_BODIES = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
}


ISSUE_ONLY_PATTERN = re.compile(
    r"^\s*(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)\s+#?\d+\s*$",
    flags=re.IGNORECASE,
)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load the input dataset from a CSV or Excel file.

    Parameters
    ----------
    filepath : str
        Path to the input dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    ValueError
        If the input file extension is not supported.
    """

    path = Path(filepath)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_text(value) -> str:
    """
    Normalize a textual value from the dataset.

    Missing values are converted to an empty string, and all other values are
    converted to strings with surrounding whitespace removed.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def classify_body(body: str) -> str | None:
    """
    Classify whether a pull request body should be excluded.

    A body is rejected when it is empty or null-like, contains only an
    issue-closing reference, or has fewer than the minimum number of words.

    Parameters
    ----------
    body : str
        Pull request body text.

    Returns
    -------
    str | None
        Exclusion reason when the PR should be rejected, or None when it is valid.
    """

    text = normalize_text(body)

    if text.lower() in INVALID_BODIES:
        return "null_body"

    if ISSUE_ONLY_PATTERN.match(text):
        return "issue_closing_only"

    words = text.split()

    if len(words) < MIN_WORDS:
        return "too_short"

    return None


def filter_prs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into valid and rejected pull requests.

    Rejected rows receive an additional column named ``exclusion_reason``
    describing why the PR was removed from the sampling frame.
    """

    valid_rows = []
    rejected_rows = []

    for _, row in df.iterrows():

        reason = classify_body(row["body"])

        if reason is None:
            valid_rows.append(row)
        else:
            rejected = row.copy()
            rejected["exclusion_reason"] = reason
            rejected_rows.append(rejected)

    valid_df = pd.DataFrame(valid_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    return valid_df, rejected_df


def allocate_samples(
    counts: pd.Series,
    sample_size: int,
) -> dict:
    """
    Allocate the target sample size across repositories.

    The allocation first assigns a minimum number of PRs per repository when
    available. The remaining sample slots are then distributed proportionally
    according to each repository's remaining valid PR capacity.
    """

    allocation = {}

    # total_valid = counts.sum()

    # mínimo por repositório
    for repo, count in counts.items():

        allocation[repo] = min(
            count,
            MIN_PER_REPO,
        )

    remaining = sample_size - sum(allocation.values())

    if remaining <= 0:
        return allocation

    remaining_capacity = {
        repo: counts[repo] - allocation[repo]
        for repo in counts.index
    }

    total_capacity = sum(
        remaining_capacity.values()
    )

    if total_capacity == 0:
        return allocation

    proportional = {}

    for repo, capacity in remaining_capacity.items():

        if capacity <= 0:
            proportional[repo] = 0
            continue

        proportional[repo] = (
            remaining
            * capacity
            / total_capacity
        )

    extra_alloc = {
        repo: int(value)
        for repo, value in proportional.items()
    }

    assigned = sum(extra_alloc.values())

    leftovers = remaining - assigned

    fractions = sorted(
        proportional.items(),
        key=lambda x: x[1] - int(x[1]),
        reverse=True,
    )

    for repo, _ in fractions[:leftovers]:
        extra_alloc[repo] += 1

    for repo in allocation:
        allocation[repo] += extra_alloc.get(repo, 0)

    return allocation


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draw a repository-stratified random sample of pull requests.

    The number of selected PRs per repository is defined by ``allocate_samples``.
    Sampling is performed with a fixed random seed to support reproducibility.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The sampled PRs and a per-repository sampling report.
    """

    counts = (
        df.groupby("repository")
        .size()
        .sort_values(ascending=False)
    )

    allocation = allocate_samples(
        counts,
        sample_size,
    )

    sampled_parts = []

    report_rows = []

    for repo, group in df.groupby("repository"):

        n = min(
            allocation.get(repo, 0),
            len(group),
        )

        sampled = group.sample(
            n=n,
            random_state=RANDOM_SEED,
        )

        sampled_parts.append(sampled)

        report_rows.append(
            {
                "repository": repo,
                "valid_prs": len(group),
                "selected_prs": n,
            }
        )

    sampled_df = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    report_df = pd.DataFrame(
        report_rows
    ).sort_values(
        by="valid_prs",
        ascending=False,
    )

    return sampled_df, report_df


def main():
    """
    Run the sampling pipeline from the command line.

    The pipeline loads the input dataset, filters invalid PR bodies, computes
    the stratified sample, and writes the resulting Excel files to the output
    directory.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    df = load_dataframe(
        args.input
    )

    print(
        f"Original rows: {len(df):,}"
    )

    valid_df, rejected_df = filter_prs(
        df
    )

    print(
        f"Valid PRs: {len(valid_df):,}"
    )

    print(
        f"Rejected PRs: {len(rejected_df):,}"
    )

    sample_size = min(
        args.sample_size,
        len(valid_df),
    )

    sampled_df, report_df = stratified_sample(
        valid_df,
        sample_size,
    )

    sampled_df.to_excel(
        output_dir / "sampled_prs_manual_validation.xlsx",
        index=False,
    )

    rejected_df.to_excel(
        output_dir / "filtered_out_prs.xlsx",
        index=False,
    )

    report_df.to_excel(
        output_dir / "sampling_report.xlsx",
        index=False,
    )

    print()
    print("Done.")
    print(
        f"Sampled PRs: {len(sampled_df):,}"
    )


if __name__ == "__main__":
    main()