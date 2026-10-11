"""
Fact Crescendo retry pass: re-fetch URLs that returned no post text, after
rewriting AMP URLs to their canonical form and with a slower request rate.
"""
import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot; contact: satyam.sunderam@kiet.edu)"
}

result_df = pd.read_csv(os.path.join(PROJECT_DIR, "05_factcrescendo_with_original_post.csv"))
failed = result_df[result_df['original_post_text'].fillna('') == ''].copy()
print("Failed rows to retry:", len(failed))

def clean_url(url):
    return url.replace('/amp/', '/').rstrip('/') + '/'

retry_results = []
for _, row in failed.iterrows():
    url = clean_url(row['url'])
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        time.sleep(4)  # slower this time
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            embeds = soup.select("blockquote.twitter-tweet")
            texts = []
            for e in embeds:
                p = e.find("p")
                texts.append(p.get_text(" ", strip=True) if p else e.get_text(" ", strip=True))
            text = " ||| ".join(texts)
        else:
            text = ""
    except Exception:
        text = ""
    retry_results.append({"orig_url": row['url'], "clean_url": url, "status": resp.status_code if 'resp' in dir() else None, "original_post_text": text})

retry_df = pd.DataFrame(retry_results)
print("\nRecovered on retry:", (retry_df['original_post_text']!='').sum(), "/", len(retry_df))
retry_df.to_csv(os.path.join(PROJECT_DIR, "05b_factcrescendo_retry.csv"), index=False)
