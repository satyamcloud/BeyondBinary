"""
Cleaning and filtering: strip media links and other URLs, collapse whitespace,
drop rows under 20 characters, and write the final dataset.
"""
import os
import re
import pandas as pd

PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")  # folder holding the pipeline CSVs

final_df = pd.read_csv(os.path.join(PROJECT_DIR, "06_final_modeling_dataset.csv"))


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)      # remove tweet media links
    text = re.sub(r'https?://\S+', '', text)                # remove any other URLs
    text = text.replace('|||', ' ')                          # soften multi-embed separator
    text = re.sub(r'\s+', ' ', text).strip()                 # collapse whitespace
    return text

final_df['text_clean'] = final_df['text'].apply(clean_text)

# drop rows that are too short after cleaning to carry real claim content
before = len(final_df)
final_df = final_df[final_df['text_clean'].str.len() >= 20].copy()
after = len(final_df)
print(f"Dropped {before - after} degenerate rows after cleaning. Remaining: {after}")

print("\nBinary label distribution after cleaning:")
print(final_df['binary_label'].value_counts())

print("\nCleaned text length distribution:")
print(final_df['text_clean'].str.len().describe())

# save cleaned version
final_df_out = final_df[['source', 'url', 'verdict', 'binary_label', 'text_clean']].rename(columns={'text_clean': 'text'})
final_df_out.to_csv(os.path.join(PROJECT_DIR, "07_final_cleaned_dataset.csv"), index=False)
print("\nSaved to 07_final_cleaned_dataset.csv")
