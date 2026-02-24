# run the pipeline for the newly added urls . urls --> transcripts --> embeddings--> vectordb

import pandas as pd
from pathlib import Path

from src.config import PODCASTS_CSV,NEW_URLS_CSV,TRANSCRIPTS_CSV,TRANSCRIPT_EMBEDDINGS_CSV
from src.url_store import add_new_urls
from src.transcript_fetcher import fetch_all_transcripts
from src.get_embeddings import add_embeddings_to_df

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
    transcripts_df=processed_or_empty(TRANSCRIPTS_CSV)

    # 3) Find truly new URLs (not already processed)
    already_processed = set(transcripts_df["url"].dropna().astype(str).tolist()) if "url" in transcripts_df.columns else set()
    new_urls = [u for u in urls if u not in already_processed]

    print(f"Total URLs in master list: {len(urls)}")
    print(f"New URLs to process: {len(new_urls)}")

    if not new_urls:
        print("Nothing new to process. Exiting.")
        return
    
    # fetch transcripts only for new urls 
    new_transcripts_df = fetch_all_transcripts(urls=new_urls, sleep_secs=1.0, save=False)

    # Append and save transcripts
    combined = pd.concat([transcripts_df, new_transcripts_df], ignore_index=True)
    #(BASE / "data" / "processed").mkdir(parents=True, exist_ok=True)
    combined.to_csv(TRANSCRIPTS_CSV, index=False)
    print(f"Saved updated transcripts to {TRANSCRIPTS_CSV}")

    
    # combined is your dataframe with transcript_clean etc.
    # Load transcripts
    df = pd.read_csv(TRANSCRIPTS_CSV)

    # Generate embeddings (automatically chunks long transcripts)
    df = add_embeddings_to_df(df, text_col="transcript_clean")

    # Save
    df.to_csv(TRANSCRIPT_EMBEDDINGS_CSV, index=False)

    # store the embeddings in the chromadb 
    


if __name__ == "__main__":
    main()