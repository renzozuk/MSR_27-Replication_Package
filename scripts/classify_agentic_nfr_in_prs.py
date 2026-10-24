"""
Creation and assignment of AGENTIC NFRs from the "agentic" column (text of the PR agentic concern).

An Agentic NFR is a NEW quality attribute specific to agentic systems that does NOT repeat any
traditional NFR and is grouped as a "child" of exactly one of them (the group/parent).

Pipeline:
  Phase 1 - Grouping: texts with coinciding terms (TF-IDF + cosine similarity) are grouped; from each
           cluster, a few representative texts are selected.
  Phase 2 - Catalog: the LLM reads the representatives (in a few batches, sequentially) and CREATES
           the new Agentic NFRs, each with a traditional parent, definition, and creation
           justification. It should reuse already-created entries when the concept is the same
           (keeps the catalog compact).
  Phase 3 - Assignment: each PR receives 1-3 Agentic NFRs (multilabel) from the catalog, with
           justification.

Outputs (prefix = input CSV name):
  <prefix>-agentic-nfr-<model>.csv           input CSV + agentic_nfr, agentic_nfr_group, agentic_nfr_reason
  <prefix>-agentic-nfr-catalogo-<model>.csv  catalog: Agentic NFR, parent, definition, creation justification, number of PRs
  <prefix>-agentic-nfr-estado-<model>.json   Phase 2 state (allows resuming)

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python classificar_prs_agentic_nfr.py                                    (uses deepseek/deepseek-v3.2)
    python classificar_prs_agentic_nfr.py meta-llama/llama-3.3-70b-instruct  (another model)
    python classificar_prs_agentic_nfr.py deepseek/deepseek-v3.2 100         (only the first 100 rows, test)

Dependencies:
    pip install openai pandas python-dotenv scikit-learn numpy
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import (AuthenticationError, BadRequestError, NotFoundError, OpenAI,
                    PermissionDeniedError)
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "annotations.csv"
TEXT_COLUMN = "agentic"
load_dotenv()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-v3.2"
TEST = int(sys.argv[2]) if len(sys.argv) > 2 else None

suffix = re.sub(r"[^a-zA-Z0-9]+", "_", MODEL).strip("_")
tag = suffix + (f"-teste{TEST}" if TEST else "")
_base = f"{INPUT_FILE.removesuffix('.csv')}-agentic-nfr"
OUTPUT_FILE = f"{_base}-{tag}.csv"
CATALOG_FILE = f"{_base}-catalogo-{tag}.csv"
STATE_FILE = f"{_base}-estado-{tag}.json"

TRADITIONAL_NFRS = ["performance", "security", "reliability", "maintainability",
                    "usability", "compatibility", "observability"]

CLUSTERING_THRESHOLD = 0.93
MAX_PER_CLUSTER = 6
TEXTS_PER_BATCH = 30
CATALOG_TARGET = 40
MAX_ENTRIES_PER_BATCH = 10
MAX_TOKENS_CATALOG = 3000
MAX_LABELS = 3
MAX_TOKENS = 300
N_WORKERS = 8
SAVE_EVERY = 16

ATTEMPTS = 3
TIMEOUT = 120
DISABLE_REASONING = True
REQUIRE_PARAMETERS = True
PROVIDER = None

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("Defina a variável de ambiente OPENROUTER_API_KEY.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=TIMEOUT,
    max_retries=3,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "PR agentic NFR builder",
    },
)

PROMPT_CATALOG = """You are a software quality researcher studying agentic AI systems (LLM-based multi-agent frameworks and workflows).

CONTEXT
Each text below describes the agentic concern of one pull request (PR). Texts are grouped by textual
similarity (shared terms); each group lists a few representative texts of a larger cluster of PRs.

TRADITIONAL NFRs (the PARENT groups): {traditional_nfrs}.

TASK
Build a catalog of AGENTIC NFRs: NEW, specific quality attributes that matter in agentic systems and
that go beyond the traditional NFRs. Every agentic NFR is a child of exactly one traditional NFR (its
parent group). Several agentic NFRs may share the same parent.
Format example (illustrative only, do not copy): "tool-call-validity" under reliability;
"agent-state-isolation" under security.

RULES
1. name: lowercase kebab-case, 2-4 words, specific to what the texts describe. It MUST NOT be the name
   of a traditional NFR.
2. Prefer REUSE: if an existing catalog entry already captures the concern, return it with
   is_new=false and its exact name. Create a new entry (is_new=true) only when no existing one fits.
   Keep the catalog compact (aim for at most {catalog_target} entries in total; it has {n} now).
3. parent_nfr: the single traditional NFR under which the entry is grouped.
4. justification (max 40 words): for a NEW entry, why it is needed (what it captures that its parent
   alone does not) and why it belongs under that parent, based on evidence in the texts; for a reused
   entry, why it applies to this batch.
5. definition (max 25 words): one sentence for NEW entries; empty string for reused ones.
6. Write analysis FIRST (max 40 words: the main concerns you see in the texts), then the entries
   needed to cover them (1 to {max_entries}).
7. All text in English.

CURRENT CATALOG
{catalog}

TEXTS
{groups}"""

PROMPT_ASSIGNMENT = """You are a software quality researcher studying agentic AI systems.

Each AGENTIC NFR in the catalog is a specific quality attribute of agentic systems, grouped under a
traditional NFR (shown in parentheses).

TASK
Given the agentic concern of a pull request (PR), assign the agentic NFR(s) from the catalog that the
PR addresses or affects (multi-label: 1 to {max_labels}).

RULES
- Use ONLY names from the catalog, spelled exactly.
- Pick only labels clearly supported by the text; prefer the most specific ones.
- Write reason FIRST (max 30 words), then the labels. English.

CATALOG
{catalog}

PR agentic concern: {text}"""

SCHEMA_CATALOG = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "agentic_nfrs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "justification": {"type": "string"},
                    "name": {"type": "string"},
                    "is_new": {"type": "boolean"},
                    "parent_nfr": {"type": "string", "enum": TRADITIONAL_NFRS},
                    "definition": {"type": "string"},
                },
                "required": ["justification", "name", "is_new", "parent_nfr", "definition"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "agentic_nfrs"],
    "additionalProperties": False,
}


def assignment_schema(names):
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "agentic_nfrs": {"type": "array", "items": {"type": "string", "enum": list(names)}},
        },
        "required": ["reason", "agentic_nfrs"],
        "additionalProperties": False,
    }


OUTPUT_COLUMNS = ["agentic_nfr", "agentic_nfr_group", "agentic_nfr_reason"]


def is_na(v):
    return str(v).strip().upper() in ("", "N/A", "NA", "NONE", "NAN")


df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", keep_default_na=False)
if TEXT_COLUMN not in df.columns:
    sys.exit(f'Coluna "{TEXT_COLUMN}" não encontrada. Colunas: {list(df.columns)}')

if TEST:
    df = df.head(TEST).copy()

if os.path.exists(OUTPUT_FILE) and not TEST:
    previous = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig", keep_default_na=False)
    for col in OUTPUT_COLUMNS:
        df[col] = previous[col].reindex(df.index) if col in previous.columns else None
else:
    for col in OUTPUT_COLUMNS:
        df[col] = None

for col in OUTPUT_COLUMNS:
    df[col] = df[col].astype("object")

for i in df.index:
    if is_na(df.at[i, TEXT_COLUMN]):
        for c in OUTPUT_COLUMNS:
            df.at[i, c] = "N/A"


def extract_json(text):
    if not text:
        raise ValueError("resposta vazia do modelo")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


LEVEL = 0
MAX_LEVEL = 3
LEVEL_DESC = {1: "without 'reasoning'", 2: "without require_parameters",
              3: "json_object with schema in prompt"}
_LOCK = threading.Lock()


def build_request(messages, level, schema, name):
    response_format = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    extra_body, messages_out = {}, messages
    if DISABLE_REASONING and level < 1:
        extra_body["reasoning"] = {"enabled": False}
    provider_cfg = {}
    if REQUIRE_PARAMETERS and level < 2:
        provider_cfg["require_parameters"] = True
    if PROVIDER:
        provider_cfg["order"] = PROVIDER
        provider_cfg["allow_fallbacks"] = False
    if provider_cfg:
        extra_body["provider"] = provider_cfg
    if level >= 3:
        response_format = {"type": "json_object"}
        messages_out = [{"role": "user", "content": messages[0]["content"]
                         + "\n\nReturn ONLY a JSON object (no extra text) that follows this JSON Schema:\n"
                         + json.dumps(schema)}]
    return messages_out, response_format, extra_body


def call_model(messages, schema, name, max_tokens, seed=42):
    global LEVEL
    while True:
        level = LEVEL
        messages_out, response_format, extra_body = build_request(messages, level, schema, name)
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages_out, response_format=response_format,
                temperature=0, max_tokens=max_tokens, seed=seed, extra_body=extra_body,
            )
        except (NotFoundError, BadRequestError) as e:
            with _LOCK:
                if level >= MAX_LEVEL:
                    raise
                if LEVEL == level:
                    LEVEL += 1
                    print(f"Warning ({type(e).__name__}): parameter not accepted; "
                          f"trying level {LEVEL} ({LEVEL_DESC[LEVEL]}).")


FATAL_ERRORS = (AuthenticationError, PermissionDeniedError)
MAX_CONSECUTIVE_FATAL = 5


def group_texts(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=int), X
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    try:
        cluster_model = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                                distance_threshold=CLUSTERING_THRESHOLD)
    except TypeError:
        cluster_model = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average",
                                                distance_threshold=CLUSTERING_THRESHOLD)
    return cluster_model.fit_predict(D), X


def choose_representatives(texts, X, labels):
    groups = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        centroid = np.asarray(X[idx].mean(axis=0))
        sims = np.asarray(X[idx] @ centroid.T).ravel()
        order = idx[np.argsort(-sims, kind="stable")]
        k = min(MAX_PER_CLUSTER, len(order))
        pos = sorted(set(np.linspace(0, len(order) - 1, k).round().astype(int)))
        groups.append({"size": len(idx), "texts": [texts[j] for j in order[pos]]})
    groups.sort(key=lambda g: (-g["size"], g["texts"][0]))
    return groups


def build_batches(groups):
    batches, current, n = [], [], 0
    for g in groups:
        if current and n + len(g["texts"]) > TEXTS_PER_BATCH:
            batches.append(current)
            current, n = [], 0
        current.append(g)
        n += len(g["texts"])
    if current:
        batches.append(current)
    return batches


def normalize_name(n):
    return re.sub(r"[^a-z0-9]+", "-", str(n).lower()).strip("-")


def format_catalog(catalog, with_definition=True):
    if not catalog:
        return "(empty)"
    lines = []
    for e in catalog.values():
        line = f"- {e['name']} (parent: {e['parent']})"
        if with_definition:
            line += f": {e['definition']}"
        lines.append(line)
    return "\n".join(lines)


def format_groups(batch):
    blocks = []
    for k, g in enumerate(batch, 1):
        blocks.append(f"Group {k} (cluster of {g['size']} PRs):\n"
                      + "\n".join(f"- {t}" for t in g["texts"]))
    return "\n\n".join(blocks)


def validate_batch(data, catalog, batch_idx):
    problems, new_entries, seen = [], [], set()
    entries = data.get("agentic_nfrs") or []
    if not entries:
        problems.append("no agentic NFR was returned; return at least one entry")
    for e in entries:
        name = normalize_name(e.get("name", ""))
        parent = str(e.get("parent_nfr", "")).strip().lower()
        if not name:
            problems.append("an entry has an empty name")
            continue
        if name in TRADITIONAL_NFRS:
            problems.append(f"'{name}' is a TRADITIONAL NFR; propose a NEW, more specific agentic "
                            f"NFR name instead (it may still be grouped under '{name}')")
            continue
        if name in catalog or name in seen:
            continue
        if parent not in TRADITIONAL_NFRS:
            problems.append(f"'{name}' needs parent_nfr from: {', '.join(TRADITIONAL_NFRS)}")
            continue
        definition = str(e.get("definition", "")).strip()
        justification = str(e.get("justification", "")).strip()
        if not definition or not justification:
            problems.append(f"'{name}' is not in the catalog, so it needs a definition and a "
                            "justification (or reuse an existing name exactly)")
            continue
        seen.add(name)
        new_entries.append({"name": name, "parent": parent, "definition": definition,
                            "justification": justification, "batch": batch_idx})
    return new_entries, problems


def process_batch(batch, catalog, batch_idx):
    base = PROMPT_CATALOG.format(
        traditional_nfrs=", ".join(TRADITIONAL_NFRS), catalog_target=CATALOG_TARGET, n=len(catalog),
        max_entries=MAX_ENTRIES_PER_BATCH, catalog=format_catalog(catalog),
        groups=format_groups(batch))
    last, problems = None, []
    for t in range(ATTEMPTS):
        prompt = base
        if problems:
            prompt += ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED. Fix these problems:\n- "
                       + "\n- ".join(problems))
        try:
            resp = call_model([{"role": "user", "content": prompt}], SCHEMA_CATALOG,
                              "agentic_nfr_catalog", MAX_TOKENS_CATALOG, seed=42 + t)
            data = extract_json(resp.choices[0].message.content)
        except FATAL_ERRORS:
            raise
        except Exception as e:
            last, problems = str(e), []
            time.sleep(2 ** t)
            continue
        new_entries, problems = validate_batch(data, catalog, batch_idx)
        if not problems:
            return new_entries
        last = "; ".join(problems)
    raise ValueError(last)


def load_state():
    if TEST or not os.path.exists(STATE_FILE):
        return {"catalogo": [], "lotes_feitos": []}
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def build_catalog(texts):
    state = load_state()
    catalog = {e["name"]: e for e in state["catalogo"]}

    labels, X = group_texts(texts)
    groups = choose_representatives(texts, X, labels)
    batches = build_batches(groups)
    n_sing = sum(1 for g in groups if g["size"] == 1)
    print(f"Phase 1: {len(texts)} texts -> {len(groups)} clusters ({n_sing} isolated) "
          f"-> {len(batches)} batches")

    failures = 0
    for k, batch in enumerate(batches, 1):
        key = hashlib.md5("\n".join(sorted(t for g in batch for t in g["texts"]))
                          .encode("utf-8")).hexdigest()
        if key in state["lotes_feitos"]:
            continue
        try:
            new_entries = process_batch(batch, catalog, k)
        except FATAL_ERRORS:
            raise
        except Exception as e:
            failures += 1
            print(f"Batch {k}/{len(batches)} failed: {e}")
            continue
        for e in new_entries:
            catalog[e["name"]] = e
        state["catalogo"] = list(catalog.values())
        state["lotes_feitos"].append(key)
        save_state(state)
        print(f"Batch {k}/{len(batches)}: +{len(new_entries)} new | catalog: {len(catalog)}")
    return catalog, failures


def classify_pr(row, catalog, schema, attempt):
    prompt = PROMPT_ASSIGNMENT.format(
        max_labels=MAX_LABELS, catalog=format_catalog(catalog),
        text=str(row[TEXT_COLUMN]).strip())
    resp = call_model([{"role": "user", "content": prompt}], schema,
                      "agentic_nfr_assignment", MAX_TOKENS, seed=42 + attempt)
    data = extract_json(resp.choices[0].message.content)

    names = []
    for n in data.get("agentic_nfrs", []):
        n = normalize_name(n)
        if n in catalog and n not in names:
            names.append(n)
    if not names:
        raise ValueError("no valid Agentic NFR from the catalog in the response")
    names = names[:MAX_LABELS]
    groups = [catalog[n]["parent"] for n in names]
    return ";".join(names), ";".join(groups), str(data.get("reason", "")).strip()


def task(i, catalog, schema):
    last_error = None
    for t in range(ATTEMPTS):
        try:
            return i, classify_pr(df.loc[i], catalog, schema, t), None
        except FATAL_ERRORS as e:
            return i, None, e
        except Exception as e:
            last_error = e
            time.sleep(2 ** t)
    return i, None, last_error


def already_processed(i, names):
    if is_na(df.at[i, TEXT_COLUMN]):
        return True
    v = df.at[i, "agentic_nfr"]
    if not isinstance(v, str) or v in ("", "N/A", "ERRO"):
        return False
    return all(n in names for n in v.split(";"))


def save():
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def export_catalog(catalog):
    counts, examples = Counter(), defaultdict(list)
    for i in df.index:
        v = df.at[i, "agentic_nfr"]
        if not isinstance(v, str) or v in ("", "N/A", "ERRO"):
            continue
        for n in v.split(";"):
            counts[n] += 1
            if len(examples[n]) < 3:
                examples[n].append(str(df.at[i, TEXT_COLUMN]).strip())
    rows = [{"parent_nfr": e["parent"], "agentic_nfr": e["name"], "n_prs": counts[e["name"]],
             "definition": e["definition"], "creation_justification": e["justification"],
             "example_texts": " | ".join(examples[e["name"]])} for e in catalog.values()]
    cat = pd.DataFrame(rows).sort_values(["parent_nfr", "n_prs"], ascending=[True, False])
    cat.to_csv(CATALOG_FILE, index=False, encoding="utf-8-sig")

    print(f"\nCatalog saved to {CATALOG_FILE}\nAgentic NFRs by group (number of PRs):")
    for parent, sub in cat.groupby("parent_nfr", sort=True):
        print(f"  {parent}")
        for _, r in sub.iterrows():
            print(f"    - {r['agentic_nfr']} ({r['n_prs']})")


if __name__ == "__main__":
    texts = list(dict.fromkeys(str(t).strip() for t in df[TEXT_COLUMN] if not is_na(t)))
    print(f"Model: {MODEL} | output: {OUTPUT_FILE}")
    print(f"{len(texts)} distinct texts in {int((~df[TEXT_COLUMN].map(is_na)).sum())} PRs "
          f"with agentic concern (of {len(df)} rows)")
    if not texts:
        sys.exit("No agentic text to process.")

    try:
        catalog, failures = build_catalog(texts)
    except FATAL_ERRORS as e:
        sys.exit(f"Authentication/permission error: {e}\nCheck key, account, and model on OpenRouter.")
    if not catalog:
        sys.exit("Empty catalog: Phase 2 did not produce any Agentic NFR.")
    if failures:
        print(f"Warning: {failures} batch(es) failed; the catalog may be incomplete. "
              "Run again to try only the missing batches.")
    print(f"Catalog with {len(catalog)} Agentic NFRs.")

    schema = assignment_schema(catalog.keys())
    pending = [i for i in df.index if not already_processed(i, set(catalog))]
    total = len(pending)
    print(f"Phase 3: {total} pending PRs | workers: {N_WORKERS}")

    start = time.time()
    consecutive_fatal = 0
    try:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(task, i, catalog, schema) for i in pending]
            for n, f in enumerate(as_completed(futures), 1):
                i, result, err = f.result()
                if err is not None:
                    print(f"Row {i}: {err}")
                    df.at[i, "agentic_nfr"] = "ERRO"
                    df.at[i, "agentic_nfr_group"] = "ERRO"
                    df.at[i, "agentic_nfr_reason"] = str(err)
                else:
                    for c, v in zip(OUTPUT_COLUMNS, result):
                        df.at[i, c] = v

                consecutive_fatal = consecutive_fatal + 1 if isinstance(err, FATAL_ERRORS) else 0
                if consecutive_fatal >= MAX_CONSECUTIVE_FATAL:
                    print(f"\n{MAX_CONSECUTIVE_FATAL} consecutive 401/403 errors: aborting. "
                          "Check key, account, and model on OpenRouter.")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break

                if n % 10 == 0 or n == total:
                    dt = time.time() - start
                    rate = n / dt
                    remaining = (total - n) / rate / 3600 if rate else 0
                    print(f"{n}/{total} | {rate:.2f} PR/s | ~{remaining:.1f} h remaining")
                if n % SAVE_EVERY == 0:
                    save()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving what was processed...")
    finally:
        save()
        print(f"Saved to {OUTPUT_FILE}")
        export_catalog(catalog)