"""
Final Dataset Merge Script
"""

import os

import pandas as pd

PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs

# --- Alt News ---
altnews = pd.read_csv(f"{PROJECT_DIR}/03_factcheck_with_original_post.csv")
altnews = altnews[altnews['verdict'].notna() & (altnews['verdict'].astype(str).str.strip() != '')]
altnews = altnews[altnews['original_post_text'].fillna('') != '']
altnews['source'] = 'altnews'
altnews['binary_label'] = altnews['verdict'].apply(
    lambda v: 'False' if v == 'False' else 'Not-Clearly-False'
)
altnews_final = altnews[['source', 'url', 'verdict', 'binary_label', 'original_post_text']].copy()
print("Alt News usable rows:", len(altnews_final))
print(altnews_final['binary_label'].value_counts())

# --- Fact Crescendo: original pass ---
fc_main = pd.read_csv(f"{PROJECT_DIR}/05_factcrescendo_with_original_post.csv")
fc_main_usable = fc_main[fc_main['original_post_text'].fillna('') != ''].copy()

# --- Fact Crescendo: retry pass (recovered failures) ---
try:
    fc_retry = pd.read_csv(f"{PROJECT_DIR}/05b_factcrescendo_retry.csv")
    fc_retry_usable = fc_retry[fc_retry['original_post_text'].fillna('') != ''].copy()
    # need verdict from the original fc_main row (retry df only has url/text)
    fc_retry_usable = fc_retry_usable.merge(
        fc_main[['url', 'verdict']].rename(columns={'url': 'orig_url'}),
        on='orig_url', how='left'
    )
    fc_retry_usable = fc_retry_usable.rename(columns={'orig_url': 'url'})
    fc_retry_usable = fc_retry_usable[['url', 'verdict', 'original_post_text']]
except FileNotFoundError:
    fc_retry_usable = pd.DataFrame(columns=['url', 'verdict', 'original_post_text'])
    print("No retry file found, skipping.")

fc_main_usable = fc_main_usable[['url', 'verdict', 'original_post_text']]

fc_combined = pd.concat([fc_main_usable, fc_retry_usable], ignore_index=True)
fc_combined = fc_combined.drop_duplicates(subset='url')
fc_combined['source'] = 'factcrescendo'
fc_combined['binary_label'] = 'Not-Clearly-False'  # scoped extraction was minority-verdict only
fc_final = fc_combined[['source', 'url', 'verdict', 'binary_label', 'original_post_text']].copy()
print("\nFact Crescendo usable rows:", len(fc_final))

# --- Merge both ---
final_df = pd.concat([altnews_final, fc_final], ignore_index=True)

# dedup: same URL should be unique already (different sites), but also
# check for duplicate original_post_text (same viral post fact-checked
# by both outlets independently)
before = len(final_df)
final_df = final_df.drop_duplicates(subset='url')
final_df = final_df.drop_duplicates(subset='original_post_text')
after = len(final_df)
print(f"\nDeduped: {before} -> {after} rows ({before-after} removed)")

# rename for modeling clarity
final_df = final_df.rename(columns={'original_post_text': 'text'})
final_df = final_df[['source', 'url', 'verdict', 'binary_label', 'text']]

print("\n=== FINAL DATASET ===")
print("Total rows:", len(final_df))
print(final_df['binary_label'].value_counts())
print("\nBy source:")
print(final_df.groupby(['source', 'binary_label']).size())

output_path = f"{PROJECT_DIR}/06_final_modeling_dataset.csv"
final_df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
