"""
Original Post Text Extraction Script
"""

import os

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs
INPUT_CSV = f"{PROJECT_DIR}/02_factcheck_with_verdict.csv"
OUTPUT_CSV = f"{PROJECT_DIR}/03_factcheck_with_original_post.csv"
CHECKPOINT_EVERY = 100

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot; contact: satyam.sunderam@kiet.edu)"
}
REQUEST_DELAY_SECONDS = 2.0


def extract_original_post_text(article_html: str) -> dict:
    soup = BeautifulSoup(article_html, "lxml")

    tweet_embeds = soup.select("blockquote.twitter-tweet")
    tweet_texts = []
    for embed in tweet_embeds:
        # twitter-tweet blockquotes typically contain a <p> with the tweet
        # text, followed by attribution links (— Author (@handle) date).
        # We take the full blockquote text but this includes the author
        # attribution line too; keep it simple and take the first <p>
        # if present (usually the actual tweet content), else full text.
        p_tag = embed.find("p")
        if p_tag:
            tweet_texts.append(p_tag.get_text(" ", strip=True))
        else:
            tweet_texts.append(embed.get_text(" ", strip=True))

    insta_embeds = soup.select("blockquote.instagram-media")

    return {
        "num_tweet_embeds": len(tweet_embeds),
        "num_insta_embeds": len(insta_embeds),
        "original_post_text": " ||| ".join(tweet_texts),  # separator between multiple embeds
    }


def main():
    df = pd.read_csv(INPUT_CSV)
    fc_df = df[df["verdict"].notna() & (df["verdict"].astype(str).str.strip() != "")].copy()
    print(f"Loaded {len(df)} total rows, {len(fc_df)} have a non-null verdict to process.")

    try:
        done_df = pd.read_csv(OUTPUT_CSV)
        done_urls = set(done_df["url"])
        print(f"Found existing progress: {len(done_urls)} rows already processed.")
        results = done_df.to_dict("records")
    except FileNotFoundError:
        done_urls = set()
        results = []

    remaining = fc_df[~fc_df["url"].isin(done_urls)]
    print(f"Remaining to process: {len(remaining)}")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        url = row["url"]
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            time.sleep(REQUEST_DELAY_SECONDS)
            if resp.status_code == 200:
                extracted = extract_original_post_text(resp.text)
            else:
                print(f"[SKIP] {url} returned {resp.status_code}")
                extracted = {"num_tweet_embeds": 0, "num_insta_embeds": 0, "original_post_text": ""}
        except requests.RequestException as e:
            print(f"[ERROR] {url}: {e}")
            extracted = {"num_tweet_embeds": 0, "num_insta_embeds": 0, "original_post_text": ""}

        row_dict = row.to_dict()
        row_dict.update(extracted)
        results.append(row_dict)

        if i % 20 == 0:
            has_text = "YES" if extracted["original_post_text"] else "no"
            print(f"  Processed {i}/{len(remaining)} — {url} -> "
                  f"tweets={extracted['num_tweet_embeds']} has_text={has_text}")

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"[CHECKPOINT] Saved {len(results)} rows to {OUTPUT_CSV}")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved {len(results)} rows to {OUTPUT_CSV}")

    final_df = pd.DataFrame(results)
    has_text = (final_df["original_post_text"].fillna("") != "").sum()
    print(f"\nRows with usable original_post_text: {has_text} / {len(final_df)} "
          f"({has_text/len(final_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
