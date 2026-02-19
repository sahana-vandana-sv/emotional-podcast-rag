# Loads the existing urls and checks if new urls matches them 
# ingestion list management 

import pandas as pd

def add_new_urls(existing_csv,new_csv)->pd.DataFrame:
    if existing_csv.exists():
        df_existing=pd.read_csv(existing_csv)
    else:
        df_existing = pd.DataFrame(columns=["urls"])

    #load new urls 
    df_new = pd.read_csv(new_csv)

    # Clean whitespace
    df_existing["urls"] = df_existing["urls"].astype(str).str.strip()
    df_new["urls"] = df_new["urls"].astype(str).str.strip()

    # Combine
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Remove duplicates
    df_combined = df_combined.drop_duplicates(subset=["urls"]).reset_index(drop=True)

    # Save back
    df_combined.to_csv(existing_csv, index=False)

    print(f"Total URLs now: {len(df_combined)}")
    return df_combined







