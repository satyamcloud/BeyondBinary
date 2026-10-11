"""
Verdict Enrichment Script
"""

import os

import json
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs
INPUT_CSV = f"{PROJECT_DIR}/01_factcheck_raw.csv"
OUTPUT_CSV = f"{PROJECT_DIR}/02_factcheck_with_verdict.csv"
CHECKPOINT_EVERY = 100  # save progress every N rows, in case of disconnect

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot; contact: satyam.sunderam@kiet.edu)"
}
REQUEST_DELAY_SECONDS = 2.0


def extract_verdict(article_html: str) -> dict:
    soup = BeautifulSoup(article_html, "lxml")
    verdict = ""
    claim_text = ""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue

        candidates = []
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            candidates = payload.get("@graph", [payload])

        for item in candidates:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "ClaimReview":
                claim_text = item.get("claimReviewed", "")
                rating = item.get("reviewRating", {})
                if isinstance(rating, dict):
                    verdict = (rating.get("alternateName")
                               or rating.get("ratingValue") or "")
                break
        if verdict:
            break
    return {"verdict": verdict, "claim_text": claim_text}


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Resume support: if output already exists, only process rows not yet done
    try:
        done_df = pd.read_csv(OUTPUT_CSV)
        done_urls = set(done_df["url"])
        print(f"Found existing progress: {len(done_urls)} rows already enriched.")
        results = done_df.to_dict("records")
    except FileNotFoundError:
        done_urls = set()
        results = []

    remaining = df[~df["url"].isin(done_urls)]
    print(f"Remaining to process: {len(remaining)}")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        url = row["url"]
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            time.sleep(REQUEST_DELAY_SECONDS)
            if resp.status_code == 200:
                extracted = extract_verdict(resp.text)
            else:
                print(f"[SKIP] {url} returned {resp.status_code}")
                extracted = {"verdict": "", "claim_text": ""}
        except requests.RequestException as e:
            print(f"[ERROR] {url}: {e}")
            extracted = {"verdict": "", "claim_text": ""}

        row_dict = row.to_dict()
        row_dict.update(extracted)
        results.append(row_dict)

        if i % 20 == 0:
            print(f"  Processed {i}/{len(remaining)} — {url} -> "
                  f"verdict='{extracted['verdict']}'")

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"[CHECKPOINT] Saved {len(results)} rows to {OUTPUT_CSV}")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
