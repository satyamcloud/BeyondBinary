"""
Fact Crescendo Original Post Extraction
"""

import os

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs
INPUT_CSV = f"{PROJECT_DIR}/factcrescendo_minority_urls.csv"
OUTPUT_CSV = f"{PROJECT_DIR}/05_factcrescendo_with_original_post.csv"
CHECKPOINT_EVERY = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot; contact: satyam.sunderam@kiet.edu)"
}
REQUEST_DELAY_SECONDS = 2.0


def extract_original_post_text(article_html: str) -> dict:
    soup = BeautifulSoup(article_html, "lxml")
    tweet_embeds = soup.select("blockquote.twitter-tweet")
    tweet_texts = []
    for embed in tweet_embeds:
        p_tag = embed.find("p")
        tweet_texts.append(p_tag.get_text(" ", strip=True) if p_tag
                            else embed.get_text(" ", strip=True))
    return {
        "num_tweet_embeds": len(tweet_embeds),
        "original_post_text": " ||| ".join(tweet_texts),
    }


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} Fact Crescendo minority-verdict rows to process.")

    try:
        done_df = pd.read_csv(OUTPUT_CSV)
        done_urls = set(done_df["url"])
        print(f"Resuming: {len(done_urls)} already processed.")
        results = done_df.to_dict("records")
    except FileNotFoundError:
        done_urls = set()
        results = []

    remaining = df[~df["url"].isin(done_urls)]
    print(f"Remaining: {len(remaining)}")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        url = row["url"]
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            time.sleep(REQUEST_DELAY_SECONDS)
            if resp.status_code == 200:
                extracted = extract_original_post_text(resp.text)
            else:
                print(f"[SKIP] {url} -> {resp.status_code}")
                extracted = {"num_tweet_embeds": 0, "original_post_text": ""}
        except requests.RequestException as e:
            print(f"[ERROR] {url}: {e}")
            extracted = {"num_tweet_embeds": 0, "original_post_text": ""}

        row_dict = row.to_dict()
        row_dict.update(extracted)
        results.append(row_dict)

        if i % 10 == 0:
            has_text = "YES" if extracted["original_post_text"] else "no"
            print(f"  {i}/{len(remaining)} -> embeds={extracted['num_tweet_embeds']} has_text={has_text}")

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"[CHECKPOINT] Saved {len(results)} rows")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    final_df = pd.DataFrame(results)
    usable = (final_df["original_post_text"].fillna("") != "").sum()
    print(f"\nDone. {len(final_df)} rows saved. Usable: {usable} ({usable/len(final_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
