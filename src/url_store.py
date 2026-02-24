# Loads the existing urls and checks if new urls matches them 
# ingestion list management 

import pandas as pd
from pathlib import Path



def add_new_urls(existing_csv:Path,new_csv:Path)->pd.DataFrame:
    if existing_csv.exists():
        df_existing = pd.read_csv(existing_csv)
        df_existing = df_existing.loc[:, ~df_existing.columns.str.match(r"^Unnamed")]
    else:
        df_existing = pd.DataFrame(columns=["url", "Topic"])

    # Enforce required columns (do NOT keep unknown columns like 'Unnamed: 0')
    for col in ["url", "Topic"]:
        if col not in df_existing.columns:
            df_existing[col] = pd.NA
    df_existing = df_existing[["url", "Topic"]]

    #load new urls 
    df_new = pd.read_csv(new_csv)
    df_new = df_new.loc[:, ~df_new.columns.str.match(r"^Unnamed")]
    print(f"Total number of new urls :{len(df_new)}")

    # Clean whitespace
    df_existing["url"] = df_existing["url"].astype(str).str.strip()
    df_new["url"] = df_new["url"].astype(str).str.strip()

    # Combine
    before_existing = len(df_existing)
    before_new = len(df_new)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Remove duplicates
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    after_dedup = len(df_combined)

    # Save back
    df_combined.to_csv(existing_csv, index=False)

    duplicates_removed = before_dedup - after_dedup
    print(f"Existing URLs loaded: {before_existing}")
    print(f"New URLs loaded (valid): {before_new}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Total URLs now: {after_dedup}")
    return df_combined







