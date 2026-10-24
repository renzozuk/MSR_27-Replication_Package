"""
Classification of PRs regarding the presence of AGENTIC CONCERN using OpenRouter (OpenAI-compatible API).

For each PR, the model decides whether it is agentic or not-agentic, with a justification
and, when agentic, a description of the affected agentic characteristics.
It uses the same input .csv as the NFR classifier and generates its own output CSV.

The key is read from .env (OPENROUTER_API_KEY) and sent in the Authorization header.

.env:
    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_REFERER=https://seu-site    (optional)
    OPENROUTER_TITLE=PR Agentic Classifier (optional)

Uso:
    python classify_agentic_text_in_prs.py                                  # default model (DeepSeek V3.2)
    python classify_agentic_text_in_prs.py deepseek/deepseek-v3.2 100       # only 100 lines (test)
    python classify_agentic_text_in_prs.py anthropic/claude-3.5-haiku

Dependências:
    pip install requests pandas python-dotenv
"""
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dotenv import load_dotenv

INPUT_FILE = "AutoGen/pulls.csv"
load_dotenv()
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-v3.2"
TEST = int(sys.argv[2]) if len(sys.argv) > 2 else None

OUTPUT_FILE = f"{INPUT_FILE.removesuffix('.csv')}-agentic.csv"

BODY_LIMIT = 1500
SAVE_EVERY = 16
N_WORKERS = 8
MAX_TOKENS = 400
ATTEMPTS = 3
HTTP_ATTEMPTS = 12
TIMEOUT = 120

INITIAL_RPM = 120
MAX_RPM = 1200
MIN_RPM = 6

AGENTIC = "agentic"
NON_AGENTIC = "not-agentic"

API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip().strip("\"'")
if not API_KEY:
    sys.exit("Defina OPENROUTER_API_KEY no .env.")
if API_KEY.startswith("{") or API_KEY.endswith("}"):
    sys.exit("OPENROUTER_API_KEY parece um placeholder (com chaves). "
             "Coloque o token puro no .env.")

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://localhost"),
    "X-Title": os.getenv("OPENROUTER_TITLE", "PR Agentic Classifier"),
}

PROMPT = """You are a software engineer reviewing a pull request (PR).

DEFINITION
- Agentic: "Agentic AI systems represent an emergent class of intelligent architectures in which
  multiple specialized agents collaborate to achieve complex, high-level objectives utilizing
  collaborative reasoning and multi-step planning."
  A PR is AGENTIC if the change adds, modifies, fixes or otherwise affects any of these:
  multi-agent coordination/orchestration, inter-agent communication, task decomposition or
  multi-step planning, tool/function calling by an agent, agent memory or state, agent reasoning
  loops, or the agent runtime/workflow engine. Judge by what the change does, not by the
  repository name. Typo fixes, CI changes and unrelated dependency bumps are not-agentic.
  If the PR touches agent behavior even indirectly, prefer "agentic".

TASK
Decide whether this PR is "agentic" or "not-agentic".

OUTPUT RULES
Write agentic_reason FIRST (max 25 words), then give the label that follows from it.
All free-text fields must be in English. agentic_text: max 30 words.

FIELDS
- agentic_reason: a real justification for the decision. Never "N/A".
- is_agentic: "agentic" or "not-agentic".
- agentic_text: if agentic, a concise description of the agentic features the PR adds, changes
  or affects; if not-agentic, exactly "N/A".

Title: {title}
Body: {body}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "agentic_reason": {"type": "string"},
        "is_agentic": {"type": "string", "enum": [AGENTIC, NON_AGENTIC]},
        "agentic_text": {"type": "string"},
    },
    "required": ["agentic_reason", "is_agentic", "agentic_text"],
    "additionalProperties": False,
}
FIELDS = tuple(SCHEMA["required"])

OUTPUT_COLUMNS = ["is_agentic", "agentic_text", "agentic_reason"]

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


class RateLimiter:
    def __init__(self, initial_rpm, min_rpm, max_rpm):
        self.min_interval = 60.0 / max_rpm
        self.max_interval = 60.0 / min_rpm
        self.interval = 60.0 / initial_rpm
        self.next_time = 0.0
        self.last_cut = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_time)
            self.next_time = start + self.interval
        if start > now:
            time.sleep(start - now)

    def penalize(self):
        with self.lock:
            now = time.monotonic()
            if now - self.last_cut >= 2.0:
                self.interval = min(self.interval * 1.5, self.max_interval)
                self.last_cut = now
                self.next_time = max(self.next_time, now + self.interval)

    def reward(self):
        with self.lock:
            self.interval = max(self.interval * 0.98, self.min_interval)

    @property
    def rpm(self):
        return 60.0 / self.interval


RATE_LIMITER = RateLimiter(INITIAL_RPM, MIN_RPM, MAX_RPM)


class FatalError(Exception):
    pass


class RetryableError(Exception):
    pass


def extract_json(text):
    if not text:
        raise ValueError("resposta vazia do modelo")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"sem JSON na resposta: {text[:150]!r}")
        return json.loads(m.group(0))


# Níveis de degradação (sobe automaticamente em caso de HTTP 400):
#   0 = response_format json_schema estrito (só provedores que suportam)
#   1 = response_format json_object + schema no prompt
#   2 = sem response_format, só schema no prompt
MAX_LEVEL = 2
LEVEL = 1
LEVEL_DESC = {1: "json_object + schema in prompt",
              2: "no response_format, schema in prompt"}
_LOCK = threading.Lock()
_LOCAL = threading.local()


def session():
    if not hasattr(_LOCAL, "s"):
        _LOCAL.s = requests.Session()
    return _LOCAL.s


def build_request(prompt, level):
    text = prompt
    if level >= 1:
        text += ("\n\nReturn ONLY a JSON object (no extra text, no code fences) with exactly "
                 "these keys, in this order: agentic_reason, is_agentic, agentic_text. "
                 "It must follow this JSON Schema:\n" + json.dumps(SCHEMA))

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        # Desliga o "thinking" do DeepSeek V3.2 (mais barato/rápido; ignorado por modelos sem reasoning)
        "reasoning": {"enabled": False},
    }
    if level == 0:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "agentic_classification", "strict": True, "schema": SCHEMA},
        }
        body["provider"] = {"require_parameters": True}
    elif level == 1:
        body["response_format"] = {"type": "json_object"}
    return body


def error_message(r):
    try:
        err = r.json().get("error", {})
        if isinstance(err, dict):
            return str(err.get("message", r.text))[:300]
        return str(err)[:300]
    except ValueError:
        return r.text[:300]


def backoff_sleep(attempt, retry_after=None):
    if retry_after:
        try:
            return min(60.0, float(retry_after)) + random.random()
        except ValueError:
            pass
    return min(30.0, 2 ** min(attempt, 5)) + random.random()


def call_model(prompt):
    global LEVEL
    http_attempt = 0
    while True:
        level = LEVEL
        body = build_request(prompt, level)
        RATE_LIMITER.wait()
        try:
            r = session().post(URL, headers=HEADERS, json=body, timeout=TIMEOUT)
        except (requests.ConnectionError, requests.Timeout):
            http_attempt += 1
            if http_attempt >= HTTP_ATTEMPTS:
                raise
            time.sleep(backoff_sleep(http_attempt))
            continue

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                raise RetryableError(f"corpo não-JSON: {r.text[:150]!r}")
            # OpenRouter pode devolver 200 com {"error": {...}} em falhas do provedor
            if isinstance(data, dict) and data.get("error") and not data.get("choices"):
                http_attempt += 1
                if http_attempt >= HTTP_ATTEMPTS:
                    raise RuntimeError(f"erro do provedor: {str(data['error'])[:300]}")
                time.sleep(backoff_sleep(http_attempt))
                continue
            RATE_LIMITER.reward()
            return data

        msg = error_message(r)
        # 401 chave inválida | 402 sem créditos | 403 moderação/chave sem permissão | 404 modelo/endpoint inexistente
        if r.status_code in (401, 402, 403, 404):
            raise FatalError(f"HTTP {r.status_code}: {msg}")

        if r.status_code == 400:
            with _LOCK:
                if level >= MAX_LEVEL:
                    raise RuntimeError(f"HTTP 400: {msg}")
                if LEVEL == level:
                    LEVEL += 1
                    print(f"Warning (HTTP 400): {msg[:120]}\n"
                          f"  -> trying level {LEVEL} ({LEVEL_DESC[LEVEL]}).")
            continue

        if r.status_code == 429:
            RATE_LIMITER.penalize()
            http_attempt += 1
            if http_attempt >= HTTP_ATTEMPTS:
                raise RuntimeError(f"HTTP 429: {msg}")
            time.sleep(backoff_sleep(http_attempt, r.headers.get("Retry-After")))
            continue

        if r.status_code in (408, 500, 502, 503, 504, 524, 529):
            http_attempt += 1
            if http_attempt >= HTTP_ATTEMPTS:
                raise RuntimeError(f"HTTP {r.status_code}: {msg}")
            time.sleep(backoff_sleep(http_attempt))
            continue

        raise RuntimeError(f"HTTP {r.status_code}: {msg}")


def extract_response(response):
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"resposta sem choices: {str(response)[:150]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # alguns provedores devolvem lista de blocos
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return extract_json(content)


def unwrap(d):
    if isinstance(d, dict) and not all(k in d for k in FIELDS):
        for v in d.values():
            if isinstance(v, dict) and all(k in v for k in FIELDS):
                return v
    return d


def classify_pr(row):
    body = row["body"] if isinstance(row["body"], str) else ""
    response = call_model(PROMPT.format(title=row["title"], body=body[:BODY_LIMIT]))
    d = unwrap(extract_response(response))

    if not isinstance(d, dict) or not all(k in d for k in FIELDS):
        raise ValueError(f"resposta sem os campos esperados: {str(d)[:150]}")

    is_agentic = d["is_agentic"]
    if is_agentic not in (AGENTIC, NON_AGENTIC):
        raise ValueError(f"rótulo inválido: {is_agentic!r}")

    agentic_reason = str(d["agentic_reason"]).strip()
    agentic_text = str(d["agentic_text"]).strip()

    if is_agentic == NON_AGENTIC:
        agentic_text = "N/A"
    elif not agentic_text or agentic_text.upper() == "N/A":
        agentic_text = ""

    return is_agentic, agentic_text, agentic_reason


MAX_CONSECUTIVE_FATALS = 5


def task(i):
    last_error = None
    for t in range(ATTEMPTS):
        try:
            return i, classify_pr(df.loc[i]), None
        except FatalError as e:
            return i, None, e
        except Exception as e:
            last_error = e
            time.sleep(1 + t)
    return i, None, last_error


def already_processed(i):
    v = df.at[i, "is_agentic"]
    return isinstance(v, str) and v in (AGENTIC, NON_AGENTIC)


def save():
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    pending = [i for i in df.index if not already_processed(i)]
    total = len(pending)
    print(f"Model: {MODEL} | provider: OpenRouter | output: {OUTPUT_FILE}")
    print(f"{total} pending PRs out of {len(df)} | workers: {N_WORKERS} | "
          f"initial level: {LEVEL} | initial RPM: {INITIAL_RPM}")

    start = time.time()
    consecutive_fatals = 0
    try:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(task, i) for i in pending]
            for n, f in enumerate(as_completed(futures), 1):
                i, res, err = f.result()
                if err is not None:
                    print(f"Row {i}: {err}")
                    for c in OUTPUT_COLUMNS:
                        df.at[i, c] = "ERRO"
                    df.at[i, "agentic_reason"] = str(err)
                else:
                    for c, v in zip(OUTPUT_COLUMNS, res):
                        df.at[i, c] = v

                consecutive_fatals = consecutive_fatals + 1 if isinstance(err, FatalError) else 0
                if consecutive_fatals >= MAX_CONSECUTIVE_FATALS:
                    print(f"\n{MAX_CONSECUTIVE_FATALS} consecutive 401/402/403/404 errors: aborting. "
                          "Check the key, your OpenRouter credits, and the model slug.")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break

                if n % 10 == 0 or n == total:
                    dt = time.time() - start
                    rate = n / dt
                    remaining = (total - n) / rate / 3600 if rate else 0
                    print(f"{n}/{total} | {rate:.2f} PR/s | current limit ~{RATE_LIMITER.rpm:.0f} req/min "
                          f"| ~{remaining:.1f} h remaining")
                if n % SAVE_EVERY == 0:
                    save()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving what has been processed...")
    finally:
        save()
        print(f"Saved to {OUTPUT_FILE}")