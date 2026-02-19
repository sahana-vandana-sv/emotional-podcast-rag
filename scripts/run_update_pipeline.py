# run the pipeline for the newly added urls . urls --> transcripts --> embeddings--> vectordb

import pandas as pd
from pathlib import Path

from src.config import PODCASTS_CSV,NEW_URLS_CSV,TRANSCRIPTS_CSV,TRANSCRIPT_EMBEDDINGS_CSV
from src.url_store import add_new_urls

def processed_or_empty(path:Path)->pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    else :
        raise FileNotFoundError(
            f"Transcripts file not found :{TRANSCRIPTS_CSV}\n"
            "Extract transcripts first"
        )

def main():
    #1. add new url and merged df 
    urls_df = add_new_urls(PODCASTS_CSV,NEW_URLS_CSV)
    urls = urls_df["url"].dropna().astype(str).tolist()

    #load already processed 
    processed=processed_or_empty(TRANSCRIPTS_CSV)

    # 3) Find truly new URLs (not already processed)
    already_processed = set(processed["url"].dropna().astype(str).tolist()) if "url" in processed.columns else set()
    new_urls = [u for u in urls if u not in already_processed]

    print(f"Total URLs in master list: {len(urls)}")
    print(f"New URLs to process: {len(new_urls)}")




if __name__ == "__main__":
    main()