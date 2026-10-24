import time
import os
import re
import requests
import json
from os import path
"""
===============================================================================
GitHub Pull Request Collector
Replication Package - Data Collection Script
===============================================================================

Description
-----------
This script collects Pull Requests (PRs) from the GitHub REST API for the
repositories analyzed in this study.

The list of repositories (18 frameworks) is intentionally hardcoded in the
REPOS variable because it exactly matches the set of frameworks analyzed in
the paper. Researchers reproducing the study should use this same list unless
they intentionally wish to conduct a different experiment.

For each Pull Request, the script also retrieves:

    - Pull Request metadata
    - Issue comments
    - Review comments
    - Commits
    - Modified files

The collected data are written incrementally to disk using the JSON Lines
(JSONL) format, making the collection process memory-efficient and resilient
to interruptions.

Main Features
-------------
• Robust against network failures.
• Automatic handling of GitHub API rate limits.
• Automatic resume after interruption.
• Streaming write (one PR per line).
• Safe for very large repositories.
• Designed for sequential execution to minimize GitHub API throttling.


Output Structure
----------------
For each repository, the script creates:

    <repository_name>_prs/
        pulls.jsonl

where every line corresponds to a complete Pull Request represented as an
independent JSON object.

Example:

    microsoft_autogen_prs/
        pulls.jsonl

The JSONL format was chosen because:

    • it allows streaming writes;
    • interrupted executions can be resumed;
    • very large datasets can be processed without loading everything into RAM.

The helper function export_jsonl_to_json_array() can later be used to convert
the JSONL file into a standard JSON array if desired.
===============================================================================
"""

# List of repositories analyzed in the study.
# This list corresponds exactly to the 18 frameworks evaluated in the paper.
REPOS = [
    "ag2ai/ag2",
    "agno-agi/agno",
    "BrainBlend-AI/atomic-agents",
    "hkuds/autoagent",
    "microsoft/autogen",
    "crewAIInc/crewAI",
    "langchain-ai/deepagents",
    "embabel/embabel-agent",
    "JetBrains/koog",
    "langchain-ai/langchain",
    "langchain4j/langchain4j",
    "langchain-ai/langgraph",
    "run-llama/llama_index",
    "n8n-io/n8n",
    "pydantic/pydantic-ai",
    "microsoft/semantic-kernel",
    "huggingface/smolagents",
    "TransformerOptimus/superagi"
]

# Personal Access Token (GitHub)
# Insert a valid GitHub token to increase the API rate limit.
TOKEN = ""


USER_AGENT = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "Mozilla/5.0"
}

MAX_NETWORK_RETRIES = 5


def github_get(url):
    """
    Authenticated GET request with automatic handling of:

    - GitHub primary rate limits;
    - GitHub secondary rate limits;
    - transient network failures.

    The function retries requests whenever possible and only raises an exception
    for unrecoverable errors.
    """
    attempt = 0
    while True:
        try:
            r = requests.get(url, headers=USER_AGENT, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as err:
            attempt += 1
            if attempt > MAX_NETWORK_RETRIES:
                raise
            wait = min(2 ** attempt, 60)
            print(f"Network erros ({err}); attempt {attempt}/{MAX_NETWORK_RETRIES} in {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code in (403, 429):
            remaining = r.headers.get('X-RateLimit-Remaining')
            reset = r.headers.get('X-RateLimit-Reset')
            if remaining == '0' and reset:
                wait = max(int(reset) - int(time.time()), 1) + 2
                print(f"Rate limit reached. Waiting {wait}s reset...")
                time.sleep(wait)
                continue

            retry_after = r.headers.get('Retry-After')
            if retry_after:
                wait = int(retry_after) + 1
                print(f"Secondary rate limit. Waiting {wait}s (Retry-After)...")
                time.sleep(wait)
                continue

        if not r.ok:

            if r.status_code == 422:
                try:
                    payload = r.json()

                    errors = payload.get("errors", [])

                    for err in errors:
                        if (
                            err.get("field") == "diff"
                            and err.get("code") == "not_available"
                        ):
                            print(f"[WARN] Diff not-available for: {url}")
                            return None

                except Exception:
                    pass

            raise Exception(
                f'GitHub returned status code {r.status_code} ({r.reason}) when loading {url}.'
            )

        return r


def _parse_next_page(r):
    """Extracts the URL of the next page from the Link header, if available."""
    link = r.headers.get('link')
    if not link:
        return None
    pages = {rel: url for url, rel in re.findall(r'<(.*?)>;\s+rel=\"(.*?)\"', link)}
    return pages.get('next')


def request_all_pulls(url):
    """
    Retrieve a single page of Pull Requests.

    The enrichment phase (comments, commits, and modified files) is intentionally
    performed later by get_pulls() so that each Pull Request can be immediately
    written to disk after processing.
    """

    if url is None:
        return None, None

    print(f"Pull request URL: {url}")
    r = github_get(url)
    data = r.json()
    return data, _parse_next_page(r)


def request_all_comments(url):
    print("Fetching comments...")

    if url is None:
        return None, None

    print(f"Comments URL: {url}")
    r = github_get(url)
    return r.json(), _parse_next_page(r)

def request_all_commits(url):
    print("Fetching commits...")

    if url is None:
        return None, None

    print(f"Commit URL: {url}")
    r = github_get(url)
    return r.json(), _parse_next_page(r)


def request_all_files(url):
    print("Fetching files ...")

    if url is None:
        return None, None

    print(f"URL File: {url}")
    r = github_get(url)
    
    if r is None:
        return [], None
    return r.json(), _parse_next_page(r)


def iter_pulls(jsonl_path):
    """
    Yields Pull Requests one at a time from a .jsonl file.

    This approach is recommended for large datasets, as it avoids loading the
    entire file into memory.

    Example:

        for pull in iter_pulls():
            ...
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def export_jsonl_to_json_array(jsonl_path, json_path):
    """
    Converts pulls.jsonl into a standard pulls.json file (a valid JSON array),
    ready to be loaded with json.load().

    The conversion is performed in streaming mode, avoiding loading the entire
    dataset into memory.
    """
    if not path.exists(jsonl_path):
        print(f"Nada a exportar: {jsonl_path} não existe.")
        return

    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as fin, \
         open(json_path, 'w', encoding='utf-8') as fout:
        fout.write("[")
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if not first:
                fout.write(",")
            fout.write("\n")
            json.dump(obj, fout, ensure_ascii=False)
            first = False
            count += 1
        fout.write("\n]")

    print(f"Exported {count} PR(s) to {json_path}.")

def _load_processed_pr_numbers(jsonl_path):
    """
    Read an existing JSONL dataset and return the set of already processed
    Pull Request numbers.

    This enables automatic resume after interrupted executions.
    """

    done = set()
    if not path.exists(jsonl_path):
        return done

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: invalid line {line_num} in {jsonl_path}; ignoring.")
                continue
            number = obj.get('number')
            if number is not None:
                done.add(number)

    return done


def _enrich_pull(pull, repo):
    """Busca comentários, review comments, commits e arquivos do PR."""
    if 'review_comments_url' in pull:
        pull['review_comments'] = get_comments(pull['review_comments_url'])
    if 'comments_url' in pull:
        pull['comments'] = get_comments(pull['comments_url'])
    if 'commits_url' in pull:
        pull['commits'] = get_commits(pull['commits_url'])
    pull['files'] = get_files(
        f"https://api.github.com/repos/{repo}/pulls/{pull['number']}/files"
    )
    return pull


def get_pulls(repo, jsonl_path):
    """
    Collect all Pull Requests from a repository.

    Each Pull Request is enriched with comments, review comments, commits,
    and modified files before being immediately appended to the JSONL dataset.

    Previously collected PRs are automatically skipped.
    """

    processed = _load_processed_pr_numbers(jsonl_path)
    if processed:
        print(f"Resumption: {len(processed)} PR(s) already collected will be ignored.")

    url = f"https://api.github.com/repos/{repo}/pulls?state=all&per_page=100"
    data, next_page = request_all_pulls(url)

    # Opening the file in append ('a') mode ensures that previously collected
    # data are preserved when the script is executed again.
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        while data:
            for pull in data:
                number = pull.get('number')
                if number in processed:
                    print(f"PR #{number} already collected; skipping.")
                    continue

                _enrich_pull(pull, repo)

                # Force data to disk immediately to avoid losing the last collected PR
                # in case of unexpected interruption.
                f.write(json.dumps(pull, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

                processed.add(number)
                print(f"PR #{number} recorded.")

            if next_page is None:
                break
            data, next_page = request_all_pulls(next_page)

    print(f"Total PRs in the file: {len(processed)}")

def get_comments(url):
    all_comments = []
    data, next_page = request_all_comments(url)
    while data:
        all_comments.extend(data)
        if next_page is None:
            break
        data, next_page = request_all_comments(next_page)
    return all_comments

def get_commits(url):
    all_commits = []
    data, next_page = request_all_commits(url)
    while data:
        all_commits.extend(data)
        if next_page is None:
            break
        data, next_page = request_all_commits(next_page)
    return all_commits

def get_files(url):
    all_files = []
    data, next_page = request_all_files(url)
    while data:
        all_files.extend(data)
        if next_page is None:
            break
        data, next_page = request_all_files(next_page)
    return all_files



"""
Entry Point
-----------

The script iterates over all repositories defined in REPOS, creates one output
directory per repository, and stores the collected Pull Requests as a JSONL
dataset.

The collection can safely be interrupted and executed again. Previously
downloaded Pull Requests will be automatically detected and skipped.
"""
if __name__ == '__main__':

    for repo in REPOS:

        output_folder = f"{repo.replace('/', '_')}_prs"
        jsonl_path = os.path.join(output_folder, "pulls.jsonl")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Downloading pull requests and writing to {jsonl_path}... ")

        get_pulls(repo, jsonl_path)

    print("\nDone.")




