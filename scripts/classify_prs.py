"""
Classification of NFR (quality attributes) in PRs with OpenRouter.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."          # or put it in a .env file
    python classify_prs.py                                   # uses deepseek/deepseek-v3.2
    python classify_prs.py meta-llama/llama-3.3-70b-instruct # another model
    python classify_prs.py deepseek/deepseek-v3.2 100        # only the first 100 rows (test)

Dependencies:
    pip install openai pandas python-dotenv
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import (AuthenticationError, BadRequestError, NotFoundError, OpenAI,
                    PermissionDeniedError)

INPUT_FILE = "AutoGen/pulls.csv"
load_dotenv()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-v3.2"
TEST = int(sys.argv[2]) if len(sys.argv) > 2 else None

# One output file per model, so results from different models never get mixed
suffix = re.sub(r"[^a-zA-Z0-9]+", "_", MODEL).strip("_")
OUTPUT_FILE = f"{INPUT_FILE.removesuffix('.csv')}-nfr-{suffix}.csv"

BODY_LIMIT = 1500
SAVE_EVERY = 10
N_WORKERS = 8
MAX_TOKENS = 400
ATTEMPTS = 3
TIMEOUT = 120

DISABLE_REASONING = True     # V3.2 supports a reasoning toggle; off = faster/cheaper
REQUIRE_PARAMETERS = True    # only route to providers that support json_schema
PROVIDER = None              # e.g. ["DeepInfra"] to pin a provider (disables fallbacks)

ATTRIBUTES = [
    "performance", "security", "reliability", "maintainability", "usability",
    "compatibility", "observability",
]

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY is not set. Export it or add it to your .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=TIMEOUT,
    default_headers={  # optional: identifies your app on OpenRouter
        "HTTP-Referer": "https://localhost",
        "X-Title": "PR NFR classification",
    },
)

NA = "N/A"

PROMPT = """You are a software engineer reviewing a pull request (PR).

TASK
Identify which non-functional requirements (NFRs), i.e. quality attributes of the software,
are addressed or affected by this PR, if any.

Quality attributes (choose from this list): performance, security, reliability, maintainability,
usability, compatibility, observability.

GUIDELINES
- A quality attribute applies when the change explicitly or evidently targets or impacts it.
  Examples: <<< mantenha aqui exatamente a mesma lista de exemplos que você já tinha >>>
- If no quality attribute fits the PR, return an empty list.
- If the title/body is vague, choose the most plausible attributes.
- A PR may affect several attributes; list all that clearly apply.

OUTPUT RULES
Write nfr_reason FIRST (max 25 words), then give the attributes that follow from it.
All free-text fields must be in English.

FIELDS
- nfr_reason: short justification for the decision.
- nfr_attributes: list of affected quality attributes; empty list if none apply.
- nfr_text: if nfr_attributes is empty, exactly "N/A". Otherwise, a concise description
  (max 30 words) of the non-functional requirement the PR addresses or affects.

Title: {title}
Body: {body}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "nfr_reason": {"type": "string"},
        "nfr_attributes": {
            "type": "array",
            "items": {"type": "string", "enum": ATTRIBUTES},
        },
        "nfr_text": {"type": "string"},
    },
    "required": ["nfr_reason", "nfr_attributes", "nfr_text"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "pr_nfr_classification", "strict": True, "schema": SCHEMA},
}

OUTPUT_COLUMNS = ["nfr_attributes", "nfr_text", "nfr_reason"]
TEXT_COLUMNS = ["nfr_attributes", "nfr_text", "nfr_reason"]

df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df["repo"] = df["url"].str.extract(r"repos/([^/]+/[^/]+)/pulls")

if TEST:
    df = df.head(TEST).copy()

if os.path.exists(OUTPUT_FILE) and not TEST:
    previous = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig", keep_default_na=False)
    for col in OUTPUT_COLUMNS:
        if col in previous.columns:
            df[col] = previous[col].reindex(df.index)
        else:
            df[col] = None
else:
    for col in OUTPUT_COLUMNS:
        df[col] = None

for col in OUTPUT_COLUMNS:
    df[col] = df[col].astype("object")


def extract_json(text):
    if not text:
        raise ValueError("empty model response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# Progressive fallback when the provider rejects some request parameter:
# 0 = full request, 1 = drop 'reasoning', 2 = drop require_parameters,
# 3 = json_object with the schema embedded in the prompt
LEVEL = 0
MAX_LEVEL = 3
LEVEL_DESC = {1: "without 'reasoning'", 2: "without require_parameters",
              3: "json_object with schema in prompt"}
_LOCK = threading.Lock()


def build_request(messages, level):
    extra, rf, msgs = {}, RESPONSE_FORMAT, messages
    if DISABLE_REASONING and level < 1:
        extra["reasoning"] = {"enabled": False}
    prov = {}
    if REQUIRE_PARAMETERS and level < 2:
        prov["require_parameters"] = True
    if PROVIDER:
        prov["order"] = PROVIDER
        prov["allow_fallbacks"] = False
    if prov:
        extra["provider"] = prov
    if level >= 3:
        rf = {"type": "json_object"}
        msgs = [{"role": "user", "content": messages[0]["content"]
                 + "\n\nReturn ONLY a JSON object (no extra text) that follows this JSON Schema:\n"
                 + json.dumps(SCHEMA)}]
    return msgs, rf, extra


def call_model(messages):
    global LEVEL
    while True:
        level = LEVEL
        msgs, rf, extra = build_request(messages, level)
        try:
            return client.chat.completions.create(
                model=MODEL, messages=msgs, response_format=rf,
                temperature=0, max_tokens=MAX_TOKENS, seed=42, extra_body=extra,
            )
        except (NotFoundError, BadRequestError) as e:
            with _LOCK:
                if level >= MAX_LEVEL:
                    raise
                if LEVEL == level:
                    LEVEL += 1
                    print(f"Warning ({type(e).__name__}): parameter not accepted; "
                          f"trying level {LEVEL} ({LEVEL_DESC[LEVEL]}).")


def classify_pr(row):
    body = row["body"] if isinstance(row["body"], str) else ""
    response = call_model([{"role": "user", "content": PROMPT.format(
        title=row["title"], body=body[:BODY_LIMIT]
    )}])
    d = extract_json(response.choices[0].message.content)

    nfr_reason = str(d["nfr_reason"]).strip()
    nfr_text = str(d["nfr_text"]).strip()
    attrs = list(dict.fromkeys(
        a for a in d.get("nfr_attributes", []) if a in ATTRIBUTES
    ))

    if not attrs:
        attrs_str = NA
        nfr_text = NA
    else:
        attrs_str = ";".join(attrs)
        if not nfr_text or nfr_text.upper() == NA:
            nfr_text = ""

    return attrs_str, nfr_text, nfr_reason


FATAL = (AuthenticationError, PermissionDeniedError)
MAX_CONSECUTIVE_FATAL = 5


def task(i):
    last_error = None
    for t in range(ATTEMPTS):
        try:
            return i, classify_pr(df.loc[i]), None
        except FATAL as e:
            return i, None, e
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(2 ** t)
    return i, None, last_error


def already_processed(i):
    v = df.at[i, "nfr_attributes"]
    return isinstance(v, str) and v not in ("", "ERROR")


def save():
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    pending = [i for i in df.index if not already_processed(i)]
    total = len(pending)
    print(f"Model: {MODEL} | output: {OUTPUT_FILE}")
    print(f"{total} PRs pending out of {len(df)} | workers: {N_WORKERS}")

    start = time.time()
    consecutive_fatal = 0
    try:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(task, i) for i in pending]
            for n, f in enumerate(as_completed(futs), 1):
                i, res, err = f.result()
                if err is not None:
                    print(f"Row {i}: {err}")
                    for c in TEXT_COLUMNS:
                        df.at[i, c] = "ERROR"
                    df.at[i, "nfr_reason"] = str(err)
                else:
                    for c, v in zip(OUTPUT_COLUMNS, res):
                        df.at[i, c] = v

                consecutive_fatal = consecutive_fatal + 1 if isinstance(err, FATAL) else 0
                if consecutive_fatal >= MAX_CONSECUTIVE_FATAL:
                    print(f"\n{MAX_CONSECUTIVE_FATAL} consecutive 401/403 errors: aborting. "
                          "Check key, account and model on OpenRouter.")
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