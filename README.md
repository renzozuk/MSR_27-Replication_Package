# Replication Package: A Mining Study of Pull Requests in Agentic AI Frameworks

This package accompanies the paper **"A Mining Study of Pull Requests in Agentic AI Frameworks"** (MSR 2027, anonymized for review). It contains the pull-request (PR) corpus, the labels produced by two human raters and an LLM, and the scripts used to collect, sample, and classify PRs.

The study mined **63,665 PRs** from eight open-source Agentic AI Frameworks (AAFs), retained **38,272** through a keyword filter, and analyzed a sample of **1,691 PRs**. Each sampled PR was labeled with seven quality categories grounded in ISO/IEC 25010:2023 (Compatibility, Maintainability, Observability, Performance, Reliability, Security, Usability) by two human raters and by an LLM (DeepSeek-V3.2). The LLM also flagged agentic concerns and synthesized 46 candidate agentic attributes.

---

## 1. Repository layout

```
.
├── LICENSE                              MIT license
├── README.md                            this file
├── dataset/
│   ├── pull_requests/                   S2: all mined PRs (63,665), one CSV per framework
│   │   ├── AutoGen/pulls.csv            3,492 PRs
│   │   ├── CrewAI/pulls.csv             3,392
│   │   ├── Embabel/pulls.csv              758
│   │   ├── Koog/pulls.csv               1,544
│   │   ├── LangChain/pulls.csv         22,154
│   │   ├── LangChain4j/pulls.csv        2,544
│   │   ├── LlamaIndex/pulls.csv         9,025
│   │   └── n8n/pulls.csv               20,756
│   └── results/
│       └── annotations.csv              S4–S9: the 1,691 sampled PRs with all labels
└── scripts/
    ├── collect_pull_requests.py         S2: GitHub PR collector
    ├── sample_prs.py                    S4: stratified sampling
    ├── classify_prs.py                  S6: LLM classification into the 7 NFR categories
    ├── classify_agentic_text_in_prs.py  S7–S8: agentic / not-agentic + description
    └── classify_agentic_nfr_in_prs.py   S9: synthesis and assignment of agentic attributes
```

The per-framework PR counts above match the "Total" column of Table I of the paper.

## 2. Mapping between paper steps and artifacts

| Step | Paper description | Artifact |
|------|-------------------|----------|
| S1 | Project selection (8 AAFs) | Listed in the paper (Section III-C); not scripted |
| S2 | Data collection (63,665 PRs) | `scripts/collect_pull_requests.py` → `dataset/pull_requests/*/pulls.csv` |
| S3 | Keyword filter (38,272 PRs) | Not included as a script (see [Section 7](#7-known-limitations-of-the-package)) |
| S4 | Sampling (1,691 PRs) | `scripts/sample_prs.py`; the resulting sample is the row set of `dataset/results/annotations.csv` |
| S5 | Human traditional NFR classification (H1, H2) | Columns `nfr_tuple (round 1)` and `nfr_tuple (round 2)` of `annotations.csv` |
| S6 | LLM traditional NFR classification | `scripts/classify_prs.py` → column `llm` |
| S7 | LLM agentic / non-agentic classification | `scripts/classify_agentic_text_in_prs.py` → column `agentic_concern` |
| S8 | LLM description of the agentic concern | same script → column `agentic_text` |
| S9 | Synthesis and assignment of agentic attributes | `scripts/classify_agentic_nfr_in_prs.py` → columns `agentic_nfr`, `agentic_nfr_group`, `agentic_nfr_reason` |

---

## 3. Dataset

### 3.1 `dataset/pull_requests/<Framework>/pulls.csv`

One row per PR, as collected from the GitHub REST API.

| Column | Description |
|--------|-------------|
| `pr_id` | Unique identifier (UUID) of the row |
| `number` | PR number in the GitHub repository |
| `title` | PR title |
| `body` | PR description (may be empty) |
| `state` | `open` or `closed` (the file does not distinguish merged from closed-unmerged PRs) |
| `url` | GitHub API URL of the PR, e.g. `https://api.github.com/repos/microsoft/autogen/pulls/7642` |

The CSV files contain only the fields used by the classification scripts. The collector script (`collect_pull_requests.py`) additionally gathers comments, review comments, commits, and modified files, and writes them to JSONL; those richer files are not part of this package.

Repositories of the eight frameworks:

| Framework | Repository |
|-----------|------------|
| AutoGen | `microsoft/autogen` |
| CrewAI | `crewAIInc/crewAI` |
| Embabel | `embabel/embabel-agent` |
| Koog | `JetBrains/koog` |
| LangChain | `langchain-ai/langchain` |
| LangChain4j | `langchain4j/langchain4j` |
| LlamaIndex | `run-llama/llama_index` |
| n8n | `n8n-io/n8n` |

### 3.2 `dataset/results/annotations.csv`

One row per sampled PR (1,691 rows, 11 columns).

| Column | Description |
|--------|-------------|
| `pr_url` | Web URL of the PR (`https://github.com/<owner>/<repo>/pull/<n>`). Unique per row |
| `pr_number` | Sequential identifier of the row in the annotation sheet. **It is not the GitHub PR number**; use `pr_url` to link to GitHub |
| `project` | Framework: `autogen`, `crewAI`, `embabel-agent`, `koog`, `langchain`, `langchain4j`, `llama_index`, `n8n` |
| `nfr_tuple (round 1)` | Categories assigned by human rater **H1**. Multi-label, separated by `\|` (e.g. `reliability\|maintainability`). Empty = PR not labeled (40 PRs) |
| `nfr_tuple (round 2)` | Categories assigned by human rater **H2**, same format. Empty = not labeled (18 PRs) |
| `llm` | Categories assigned by the LLM (DeepSeek-V3.2), same format. Empty = not labeled (1 PR) |
| `agentic_concern` | `yes` (885 PRs) or `no` (806 PRs), LLM judgment |
| `agentic_text` | LLM description of the agentic concern; `N/A` when `agentic_concern = no` |
| `agentic_nfr` | Agentic attribute(s) assigned to the PR, separated by `;`, as kebab-case identifiers (1–3 per PR); `N/A` when not agentic |
| `agentic_nfr_group` | Traditional category each assigned attribute is anchored to, `;`-separated and aligned position by position with `agentic_nfr`; `N/A` when not agentic |
| `agentic_nfr_reason` | LLM justification for the assignment; `N/A` when not agentic |

Notes on the columns:

- The category vocabulary is `compatibility`, `maintainability`, `observability`, `performance`, `reliability`, `security`, `usability`. The paper's *Reliability* corresponds to *Robustness* in Oliveira et al. [4].
- Counts of labeled PRs match the paper: 1,651 for H1, 1,673 for H2, and 1,690 for the LLM. Prevalence in Table II is computed over the PRs each rater labeled.
- Attribute identifiers are the kebab-case form of the names in Tables IV and V (e.g. `memory-state-persistence` = *Memory State Persistence*). One identifier differs in wording: `streaming-interaction-reliability` corresponds to *Streaming Reaction Reliability* in Table V.
- The 46 distinct attribute identifiers in `agentic_nfr` are exactly the 46 candidate attributes of the paper (24 anchored to Reliability). They are LLM-generated and **have not been validated by humans**.
- There is no adjudicated gold standard: the three label columns are independent evaluations.

### 3.3 Linking a sampled PR to its full record

`annotations.csv` links to `pulls.csv` through the PR URL. The API URL in `pulls.csv` can be converted to the web URL:

```python
import glob
import pandas as pd

root = "dataset"
ann = pd.read_csv(f"{root}/results/annotations.csv", keep_default_na=False)

prs = pd.concat(
    pd.read_csv(f).assign(framework=f.split("/")[-2])
    for f in glob.glob(f"{root}/pull_requests/*/pulls.csv")
)
prs["pr_url"] = (prs["url"]
                 .str.replace("https://api.github.com/repos/", "https://github.com/", regex=False)
                 .str.replace("/pulls/", "/pull/", regex=False))

merged = ann.merge(prs, on="pr_url", how="left")
assert merged["pr_id"].notna().all()   # all 1,691 sampled PRs are found
```

---

## 4. Scripts

All scripts are standalone command-line programs. Python 3.10 or newer is required (`sample_prs.py` uses `str | None` type hints).

### 4.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install pandas requests openai python-dotenv scikit-learn numpy openpyxl
```

### 4.2 Credentials

- **GitHub** (`collect_pull_requests.py`): set a personal access token in the `TOKEN` variable at the top of the script (left empty in the package). Without a token, the GitHub API rate limit makes collection impractical.
- **LLM** (the three `classify_*` scripts): the scripts call the OpenAI-compatible **OpenRouter** API. Create a `.env` file next to where you run the scripts:

  ```
  OPENROUTER_API_KEY=sk-or-v1-...
  ```

### 4.3 `collect_pull_requests.py` (S2)

Downloads every PR (`state=all`) of each repository in the `REPOS` list through the GitHub REST API, enriching each PR with issue comments, review comments, commits, and modified files. Output is written incrementally to `<owner>_<repo>_prs/pulls.jsonl` (one PR per line); the script resumes automatically after an interruption and handles primary and secondary rate limits.

```bash
python scripts/collect_pull_requests.py
```

> **Important.** The `REPOS` list in this script contains 18 repositories, a superset of the eight frameworks analyzed in the paper. To reproduce the study, keep only: `microsoft/autogen`, `crewAIInc/crewAI`, `embabel/embabel-agent`, `JetBrains/koog`, `langchain-ai/langchain`, `langchain4j/langchain4j`, `run-llama/llama_index`, and `n8n-io/n8n`. Because the repositories keep receiving PRs, a new collection will return more than 63,665 PRs; use the CSV files in `dataset/pull_requests/` to work with the exact corpus of the paper.

### 4.4 `sample_prs.py` (S4)

Removes PRs whose description is unsuitable for manual analysis (empty or null-like, only an issue-closing reference such as `Fixes #123`, or fewer than 10 words) and draws a repository-stratified random sample (at least 5 PRs per repository when available, the rest allocated in proportion to each repository's number of valid PRs; `RANDOM_SEED = 42`).

```bash
python scripts/sample_prs.py --input <keyword-filtered CSV or XLSX> --sample-size <N> [--output-dir <dir>]
```

The input must contain a `repository` column and a `body` column. Outputs: `sampled_prs_manual_validation.xlsx`, `filtered_out_prs.xlsx`, and `sampling_report.xlsx`.

> The sampling frame (the 38,272 keyword-filtered PRs) is not included in the package, and the default `--sample-size` in the script (1,046) differs from the final sample size of the paper (1,691). The sample actually used is the set of rows of `dataset/results/annotations.csv`.

### 4.5 `classify_prs.py` (S6)

Asks the LLM which of the seven quality categories each PR addresses (multi-label, possibly none), together with a short justification and a description (`nfr_attributes`, `nfr_text`, `nfr_reason`). The input is a PR CSV (title and the first 1,500 characters of the body).

```bash
python scripts/classify_prs.py                                  # default: deepseek/deepseek-v3.2
python scripts/classify_prs.py deepseek/deepseek-v3.2 100       # smoke test on 100 rows
```

The input path is set by the `INPUT_FILE` constant at the top of the script (currently `AutoGen/pulls.csv`); edit it, or loop over the frameworks, to process the other CSV files. One output file per model is written next to the input (`<input>-nfr-<model>.csv`), and an interrupted run resumes from where it stopped.

### 4.6 `classify_agentic_text_in_prs.py` (S7–S8)

Decides whether each PR is `agentic` or `not-agentic` according to the seven criteria of Section III-C (multi-agent coordination, inter-agent communication, task decomposition/multi-step planning, tool/function calling, agent memory/state, reasoning loops, agent runtime/workflow engine). For agentic PRs it also writes a short description of the agentic concern.

```bash
python scripts/classify_agentic_text_in_prs.py
python scripts/classify_agentic_text_in_prs.py deepseek/deepseek-v3.2 100   # smoke test
```

Output: `<input>-agentic.csv` with columns `is_agentic`, `agentic_text`, `agentic_reason`. As in `classify_prs.py`, the input path is the `INPUT_FILE` constant.

### 4.7 `classify_agentic_nfr_in_prs.py` (S9)

Builds the catalog of agentic attributes from the `agentic_text` descriptions and assigns attributes to each agentic PR, in three phases:

1. **Grouping.** Distinct descriptions are vectorized with TF-IDF (1–2-grams) and clustered by agglomerative clustering (average linkage, cosine distance, threshold 0.93). Up to 6 representative descriptions are selected per cluster.
2. **Catalog creation.** Representatives are sent to the LLM in sequential batches of about 30 texts. In each batch the LLM sees the current catalog and either reuses an existing attribute or creates a new one with a definition, a justification, and exactly one traditional parent category. The catalog is intended to stay compact (target of about 40 entries; a batch of at most 10 new entries). Names that coincide with a traditional category are rejected, and invalid answers are re-prompted (up to 3 attempts). Progress is stored in a JSON state file so the phase can be resumed.
3. **Assignment.** Each agentic PR receives 1–3 attributes from the final catalog (the output schema restricts the labels to catalog names), with a justification. The parent category of each attribute gives `agentic_nfr_group`.

```bash
python scripts/classify_agentic_nfr_in_prs.py                              # default model
python scripts/classify_agentic_nfr_in_prs.py deepseek/deepseek-v3.2 100   # smoke test
```

Outputs, named after the input file and the model: the input CSV plus `agentic_nfr`, `agentic_nfr_group`, `agentic_nfr_reason`; a catalog CSV (attribute, parent, definition, creation justification, number of PRs, example texts); and the JSON state file.

> The script reads the file named in `INPUT_FILE` (`annotations.csv`) and the text column named in `TEXT_COLUMN` (`agentic`). In the shipped `annotations.csv` the description column is called `agentic_text` (and the flag `agentic_concern`, with `yes`/`no` values); adapt `TEXT_COLUMN` or rename the column of the output of S7–S8 accordingly.

### 4.8 LLM configuration used by the three `classify_*` scripts

| Setting | Value |
|---------|-------|
| Model | `deepseek/deepseek-v3.2` (DeepSeek-V3.2), default argument of every script |
| Endpoint | OpenRouter, OpenAI-compatible chat-completions API |
| Temperature | 0 |
| Seed | 42 (the S9 scripts vary it across retries: 42 + attempt) |
| Reasoning ("thinking") | disabled (`reasoning: {enabled: false}`) |
| Input | PR title + first 1,500 characters of the body (`BODY_LIMIT`) |
| Output | JSON constrained by a schema; automatic fallback to `json_object` with the schema in the prompt if the provider rejects `json_schema` |
| Prompts | one prompt per task; they are string constants (`PROMPT`, `PROMPT_CATALOG`, `PROMPT_ASSIGNMENT`) at the top of each script |
| Parallelism | 8 worker threads; automatic retries, and adaptive rate limiting in the S7–S8 script |

Because requests are routed by OpenRouter to third-party providers and model versions can change, rerunning the classification can produce labels that differ from those in `annotations.csv`. The archived labels in this package are the ones the paper reports.

---

## 5. Reproducing the reported numbers

Everything in RQ1–RQ3 that depends on the labels can be recomputed from `dataset/results/annotations.csv` without any API key.

### 5.1 Quick check (Table II, Table VI, agentic share)

```python
import pandas as pd

a = pd.read_csv("dataset/results/annotations.csv", keep_default_na=False)
cats = ["compatibility", "maintainability", "observability",
        "performance", "reliability", "security", "usability"]

# Table II: share of labeled PRs per category and rater
for name, col in [("H1", "nfr_tuple (round 1)"),
                  ("H2", "nfr_tuple (round 2)"),
                  ("LLM", "llm")]:
    labeled = a[a[col].str.strip() != ""][col]
    sets = labeled.str.split("|").map(set)
    print(name, len(labeled),
          {c: round(100 * sets.map(lambda s: c in s).mean(), 2) for c in cats})

# RQ2: agentic share (885 of 1,691 = 52.34%)
agentic = a[a.agentic_concern == "yes"]
print(len(agentic), round(100 * len(agentic) / len(a), 2))

# Table VI (a): agentic PRs per framework
print(a.groupby("project").agentic_concern.apply(lambda s: (s == "yes").sum()))

# Table VI (b): agentic PRs with at least one attribute anchored to each category
print(agentic.agentic_nfr_group.str.split(";").map(set).explode().value_counts())

# Tables IV and V: PRs per agentic attribute (46 distinct attributes)
print(agentic.agentic_nfr.str.split(";").explode().value_counts())
```

Expected sizes: 1,651 / 1,673 / 1,690 labeled PRs (H1 / H2 / LLM), 885 agentic PRs, 46 distinct attributes. Output of the snippet for the LLM column: Compatibility 15.92%, Maintainability 37.93%, Observability 1.66%, Performance 9.41%, Reliability 43.55%, Security 9.17%, Usability 25.92%.

### 5.2 Agreement statistics (Table III)

Per-category Cohen's κ (rater pairs), Fleiss' κ (three raters), and Krippendorff's α with MASI and Jaccard distances (2,000-resample percentile bootstrap over PRs) are computed over the three label columns of `annotations.csv`, excluding PRs lacking a label from a rater (pairwise for pair statistics; the 1,635 PRs labeled by all three raters for three-rater statistics). Standard implementations of these coefficients can be applied directly to the three columns, e.g. `sklearn.metrics.cohen_kappa_score`, `statsmodels.stats.inter_rater.fleiss_kappa`, and `nltk.metrics.agreement.AnnotationTask` with `masi_distance`.

> The script that produced Table III is not part of this package.

### 5.3 Comparison with Spring (Table VII)

The Spring baseline numbers come from Oliveira et al. (reference [4] in the paper) and are not redistributed here. The AAF columns of Table VII follow from the category counts of Section 5.1 restricted to Maintainability, Performance, Reliability, and Security, each divided by the sum of the four counts.

---

## 6. Ethical and legal notes

- The data consist of public PR titles and descriptions from public GitHub repositories. No private data were collected, and no PR author information is included in the CSV files.
- PR text is authored by the respective contributors and remains subject to the licenses of the original repositories.
- LLM labels and agentic attributes are research annotations produced by a model; they must not be read as ground truth (see the Threats to Validity section of the paper).

## 7. Known limitations of the package

- **Keyword filter (S3).** The script and the keyword list that reduced 63,665 PRs to 38,272 are not included, nor is the list of filtered PRs.
- **Agreement analysis.** The code for Table III and for the framework-weighted estimates of Section IV-B is not included. Section 5.2 describes how to recompute the coefficients.
- **Human labeling.** Raters' guidelines and the artifacts shown to them are not included. The identity of the raters behind the columns `nfr_tuple (round 1)` and `nfr_tuple (round 2)` (H1 and H2) is documented only by the counts of labeled PRs.
- **Hard-coded paths.** `classify_prs.py` and `classify_agentic_text_in_prs.py` read `AutoGen/pulls.csv`; `classify_agentic_nfr_in_prs.py` reads `annotations.csv`. Paths and column names must be adapted before running (see Sections 4.5–4.7).
- **Prompt of `classify_prs.py`.** The guideline section of this prompt contains a placeholder (`<<< ... >>>`) where the list of examples should appear; [AUTHOR: restore the exact examples used in the study before release].
- **Non-determinism.** LLM outputs may vary with provider and model version even at temperature 0.

## 8. License

The code and data of this package are released under the MIT License (see `LICENSE`).

## 9. Citation

```
[AUTHOR: add BibTeX entry after the review period]
```