# scripts/recover_transcripts_from_chromadb.py
# -----------------------------------------------------------
# Recovery script: reconstruct transcripts_df.csv from ChromaDB
#
# Usage:
#   python scripts/recover_transcripts_from_chromadb.py
#
# The script:
#   1. Connects to the existing ChromaDB collection (podcast_episodes)
#   2. Extracts all documents, metadata, and embeddings
#   3. Reconstructs and saves data/raw/transcripts_df.csv
#
# Note on field recovery:
#   - url, video_id, youtube_title, youtube_channel, duration_mins,
#     word_count are fully recovered from ChromaDB metadata.
#   - transcript_clean and transcript_text are recovered from the
#     ChromaDB document field (stored as transcript_clean[:1000] chars).
#   - embedding is recovered from ChromaDB embeddings.
#   - raw_segments, num_segments, status, error, fetched_at are not
#     stored in ChromaDB and will be set to best-effort defaults.
# -----------------------------------------------------------

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME, TRANSCRIPTS_CSV, RAW_DATA_DIR


def recover_transcripts_from_chromadb(output_path: Path = TRANSCRIPTS_CSV) -> pd.DataFrame:
    """
    Connect to the existing ChromaDB collection, extract all stored data,
    and reconstruct transcripts_df.csv.

    Parameters
    ----------
    output_path : Path
        Where to save the recovered CSV. Defaults to data/raw/transcripts_df.csv.

    Returns
    -------
    pd.DataFrame
        The recovered DataFrame.
    """
    # ── 1. Connect to ChromaDB ───────────────────────────────────────────────
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_DIR}\n"
            "Cannot recover transcripts without the ChromaDB database."
        )

    print(f"Connecting to ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in ChromaDB.\n"
            f"Original error: {e}"
        )

    total = collection.count()
    print(f"✓ Found collection '{COLLECTION_NAME}' with {total} episodes")

    if total == 0:
        print("⚠️  Collection is empty — nothing to recover.")
        return pd.DataFrame(columns=[
            "url", "video_id", "youtube_title", "youtube_channel",
            "status", "error", "transcript_text", "transcript_clean",
            "duration_mins", "num_segments", "raw_segments",
            "word_count", "fetched_at", "embedding",
        ])

    # ── 2. Extract all data from ChromaDB ────────────────────────────────────
    print("Extracting all documents, metadata, and embeddings …")
    result = collection.get(
        include=["embeddings", "documents", "metadatas"],
    )

    ids        = result.get("ids", [])
    embeddings = result.get("embeddings", [])
    documents  = result.get("documents", [])
    metadatas  = result.get("metadatas", [])

    print(f"  Retrieved {len(ids)} records")

    # ── 3. Reconstruct rows ───────────────────────────────────────────────────
    records = []
    for i, episode_id in enumerate(ids):
        meta     = metadatas[i] if i < len(metadatas) else {}
        doc      = documents[i]  if i < len(documents)  else ""
        emb      = embeddings[i] if i < len(embeddings)  else None

        # ChromaDB metadata field names → original CSV column names
        url              = meta.get("url")
        video_id         = meta.get("video_id")
        youtube_title    = meta.get("episode_title")
        youtube_channel  = meta.get("show_name")
        duration_mins    = meta.get("duration_mins")
        word_count       = meta.get("word_count")

        # transcript_clean was stored as document (up to 1000 chars)
        transcript_clean = doc or ""
        # transcript_text is best-effort: same as transcript_clean from ChromaDB
        transcript_text  = transcript_clean

        records.append({
            "url":              url,
            "video_id":         video_id,
            "youtube_title":    youtube_title,
            "youtube_channel":  youtube_channel,
            "status":           "success",   # only successful episodes were indexed
            "error":            None,
            "transcript_text":  transcript_text,
            "transcript_clean": transcript_clean,
            "duration_mins":    duration_mins,
            "num_segments":     None,        # not stored in ChromaDB
            "raw_segments":     None,        # not stored in ChromaDB
            "word_count":       word_count,
            "fetched_at":       None,        # not stored in ChromaDB
            "embedding":        emb,
        })

    # Sort by the numeric episode_id so the output order matches the original
    def _sort_key(record_and_id):
        _, eid = record_and_id
        try:
            return int(eid)
        except (ValueError, TypeError):
            return 0

    sorted_pairs = sorted(zip(records, ids), key=_sort_key)
    records = [r for r, _ in sorted_pairs]

    df = pd.DataFrame(records, columns=[
        "url", "video_id", "youtube_title", "youtube_channel",
        "status", "error", "transcript_text", "transcript_clean",
        "duration_mins", "num_segments", "raw_segments",
        "word_count", "fetched_at", "embedding",
    ])

    # ── 4. Save to CSV ────────────────────────────────────────────────────────
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} recovered episodes to: {output_path}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n── Recovery Summary ──────────────────────────────────────")
    print(f"  Episodes recovered : {len(df)}")
    print(f"  Columns            : {df.columns.tolist()}")
    print(f"  Has embeddings     : {df['embedding'].notna().sum()} / {len(df)}")
    print(f"  Output path        : {output_path}")
    print("──────────────────────────────────────────────────────────")
    print("\n⚠️  Note: transcript_text and transcript_clean are recovered")
    print("   from the ChromaDB document field (limited to 1000 chars).")
    print("   raw_segments, num_segments, and fetched_at could not be")
    print("   recovered as they were not stored in ChromaDB.")

    return df


if __name__ == "__main__":
    recover_transcripts_from_chromadb()
