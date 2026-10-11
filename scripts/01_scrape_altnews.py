"""
Hinglish Misinformation Dataset — Fact-Check Scraper
Sources: Alt News (altnews.in), Boom Live (boomlive.in)
"""

import time
import csv
import re
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
import trafilatura

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Academic Research Bot; contact: satyam.sunderam@kiet.edu)"
}

REQUEST_DELAY_SECONDS = 2.5   # conservative rate limit — be polite
MAX_PAGES_PER_SITE = 50       # tune based on how much data you need
PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs
OUTPUT_CSV = f"{PROJECT_DIR}/01_factcheck_raw.csv"

SOURCES = {
    "altnews": {
        "base_url": "https://www.altnews.in",
        # Homepage /page/N/ is a fixed magazine layout, NOT a chronological
        # archive — it repeats the same ~6 curated items per section on
        # every page. Real chronological archives live under /type/<slug>/.
        "type_archives": {
            "fact-check": "https://www.altnews.in/type/fact-check/page/{page}/",
            "media-analysis": "https://www.altnews.in/type/media-analysis/page/{page}/",
            "investigation": "https://www.altnews.in/type/investigation/page/{page}/",
            "objectionable-speech": "https://www.altnews.in/type/objectionable-speech/page/{page}/",
            "platform-accountability": "https://www.altnews.in/type/platform-accountability/page/{page}/",
            "report": "https://www.altnews.in/type/report/page/{page}/",
        },
    },
}


# ---------------------------------------------------------------------------
# robots.txt compliance check
# ---------------------------------------------------------------------------

def can_fetch(base_url: str, target_url: str, user_agent: str = "*") -> bool:
    """Check robots.txt before crawling. Always run this before scraping."""
    rp = RobotFileParser()
    rp.set_url(urljoin(base_url, "/robots.txt"))
    try:
        rp.read()
    except Exception as e:
        print(f"[WARN] Could not read robots.txt for {base_url}: {e}")
        print("[WARN] Proceeding with EXTRA caution / manual review recommended.")
        return False
    return rp.can_fetch(user_agent, target_url)


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def polite_get(url: str) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        time.sleep(REQUEST_DELAY_SECONDS)
        if resp.status_code == 200:
            return resp
        print(f"[SKIP] {url} returned status {resp.status_code}")
        return None
    except requests.RequestException as e:
        print(f"[ERROR] {url}: {e}")
        return None


CARD_LABEL_PATTERN = re.compile(
    r"^(Fact Check(?:\s*(?:False|True|Misleading|Partly False))?|Report|"
    r"Investigation|Media Analysis|Objectionable Speech|Platform Accountability)$",
    re.IGNORECASE,
)


def extract_article_links_altnews(listing_html: str, base_url: str) -> list[dict]:
    """
    Returns list of {"url": ..., "card_label": ...} for each article card
    on a /type/<slug>/page/N/ archive page. card_label captures the small
    tag shown above each headline (e.g. "Fact Check False", "Fact Check True")
    which gives us verdict info directly from the listing, no need to guess
    it from body text later.
    """
    soup = BeautifulSoup(listing_html, "lxml")
    results = []
    seen_urls = set()

    for heading in soup.find_all(["h2", "h3"]):
        a = heading.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        if urlparse(href).netloc not in ("", urlparse(base_url).netloc):
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue

        # look for a short label element appearing just before this heading
        card_label = ""
        node = heading.find_previous(["a", "span", "div", "p"])
        hops = 0
        while node is not None and hops < 5:
            text = node.get_text(strip=True)
            if text and len(text) < 40 and CARD_LABEL_PATTERN.match(text):
                card_label = text
                break
            node = node.find_previous(["a", "span", "div", "p"])
            hops += 1

        seen_urls.add(full_url)
        results.append({"url": full_url, "card_label": card_label})

    return results


def extract_article_content(article_html: str) -> dict:
    """
    Body text via trafilatura (robust across WordPress theme variations).
    Category/verdict now comes from the archive type + card label captured
    during listing-page scraping, not from parsing this page.
    """
    soup = BeautifulSoup(article_html, "lxml")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    date = ""
    time_tag = soup.find("time")
    if time_tag:
        date = time_tag.get("datetime", time_tag.get_text(strip=True))

    body_text = trafilatura.extract(article_html, include_comments=False,
                                     include_tables=False) or ""

    return {
        "title": title,
        "date": date,
        "body_text": body_text,
    }


# ---------------------------------------------------------------------------
# Main scraping loop
# ---------------------------------------------------------------------------

def load_checkpoint(filepath: str) -> list[dict]:
    """Load existing per-type checkpoint if it exists (resume support)."""
    import os
    if not os.path.exists(filepath):
        return []
    df_rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            df_rows.append(row)
    return df_rows


def scrape_source(source_name: str, config: dict, max_pages: int) -> list[dict]:
    import os
    base_url = config["base_url"]
    all_records = []
    seen_urls_global = set()

    for type_slug, url_template in config["type_archives"].items():
        checkpoint_path = f"{PROJECT_DIR}/checkpoint_{source_name}_{type_slug}.csv"

        # RESUME: if this type was already fully scraped in a prior session,
        # load it and skip re-scraping entirely.
        existing = load_checkpoint(checkpoint_path)
        if existing:
            print(f"\n=== [{type_slug}] Found existing checkpoint "
                  f"({len(existing)} records) — skipping re-scrape, loading instead. ===")
            all_records.extend(existing)
            seen_urls_global.update(r["url"] for r in existing)
            continue

        print(f"\n=== [{source_name}] Checking robots.txt for /type/{type_slug}/ ===")
        sample_listing = url_template.format(page=1)
        allowed = can_fetch(base_url, sample_listing)
        print(f"robots.txt allows crawling {sample_listing}: {allowed}")
        if not allowed:
            print(f"[STOP] robots.txt disallows /type/{type_slug}/. Skipping this category.")
            continue

        type_records = []
        prev_page_urls = None
        for page in range(1, max_pages + 1):
            listing_url = url_template.format(page=page)
            print(f"[{type_slug}] Fetching page {page}: {listing_url}")
            resp = polite_get(listing_url)
            if resp is None:
                break

            cards = extract_article_links_altnews(resp.text, base_url)
            page_urls = {c["url"] for c in cards}

            if prev_page_urls is not None and page_urls == prev_page_urls:
                print(f"[{type_slug}] Page {page} identical to previous page — "
                      f"reached end of archive, stopping.")
                break
            prev_page_urls = page_urls

            new_cards = [c for c in cards if c["url"] not in seen_urls_global]
            if not new_cards and cards:
                print(f"[{type_slug}] All {len(cards)} cards on page {page} "
                      f"already seen — stopping pagination for this type.")
                break
            if not cards:
                print(f"[{type_slug}] No cards found on page {page} — stopping.")
                break

            for card in new_cards:
                link = card["url"]
                print(f"  -> [{card['card_label'] or type_slug}] {link}")
                seen_urls_global.add(link)
                article_resp = polite_get(link)
                if article_resp is None:
                    continue
                data = extract_article_content(article_resp.text)
                data["source"] = source_name
                data["archive_type"] = type_slug
                data["card_label"] = card["card_label"]
                data["url"] = link
                type_records.append(data)

        # CHECKPOINT: save this type's data to Drive immediately, before
        # moving to the next type. A disconnect after this point loses
        # at most one archive-type's progress, not everything.
        save_to_csv(type_records, checkpoint_path)
        print(f"[{type_slug}] Checkpoint saved: {checkpoint_path}")
        all_records.extend(type_records)

    return all_records


def save_to_csv(records: list[dict], filepath: str):
    import os
    if not records:
        print(f"No records to save for {filepath}.")
        # still write an empty file with headers so resume-check doesn't
        # mistake "genuinely zero results" for "never attempted"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fieldnames = ["source", "archive_type", "card_label", "url", "title", "date", "body_text"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["source", "archive_type", "card_label", "url", "title", "date", "body_text"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"\nSaved {len(records)} records to {filepath}")


if __name__ == "__main__":
    all_records = []
    for source_name, config in SOURCES.items():
        records = scrape_source(source_name, config, MAX_PAGES_PER_SITE)
        all_records.extend(records)

    save_to_csv(all_records, OUTPUT_CSV)
