# src/transcript_fetcher.py
# -----------------------------------------------------------
# Loads YouTube URLs, fetches transcripts, cleans them,
# and logs every error to logs/transcript_errors.log
#
# Pipeline:
#   podcast_urls.txt
#     → extract video ID from URL
#     → fetch transcript via youtube-transcript-api
#     → clean transcript text
#     → save to data/processed/transcripts_df.csv 
#     → log any failures to logs/transcript_errors.log
#
# Install dependency first:
#   pip install youtube-transcript-api
# -----------------------------------------------------------

import re
import time
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from src.config import PROCESSED_DATA_DIR ,PODCASTS_CSV,TRANSCRIPTS_CSV,LOG_DIR,LOG_FILE



# -----------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------
def _setup_logger() -> logging.Logger:
    """
    Writes to both:
      - logs/transcript_errors.log  (persistent file)
      - terminal / notebook          (StreamHandler)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("transcript_fetcher")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if cell is re-run
    if logger.handlers:
        logger.handlers.clear()

    # File handler (all levels)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)-8s | %(message)s"
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = _setup_logger()


# -----------------------------------------------------------
# STEP 1: Load URLs from file
# -----------------------------------------------------------
def load_urls(filepath: Path = PODCASTS_CSV) -> list[str]:
    """
    Read YouTube URLs from a plain text file (one URL per line).
    Lines starting with # are treated as comments and skipped.

    Parameters
    ----------
    filepath : Path to your youtube_urls.txt

    Returns
    -------
    list of clean URL strings

    Example youtube_urls.txt
    ------------------------
    # Brené Brown episodes
    https://www.youtube.com/watch?v=TbsRU-crgsc
    https://www.youtube.com/watch?v=jroF3PH-PTs
    """
    if not filepath.exists():
        logger.error(f"URLs file not found: {filepath}")
        raise FileNotFoundError(
            f"Create a file at {filepath} with one YouTube URL per line."
        )

    urls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    logger.info(f"Loaded {len(urls)} URLs from {filepath}")
    return urls


# -----------------------------------------------------------
# STEP 2: Extract video ID from URL
# -----------------------------------------------------------
def extract_video_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from any common URL format.

    Supported formats
    -----------------
    https://www.youtube.com/watch?v=TbsRU-crgsc
    https://youtu.be/TbsRU-crgsc
    https://www.youtube.com/embed/TbsRU-crgsc
    https://www.youtube.com/watch?v=TbsRU-crgsc&t=120s

    Returns None if no match found.
    """
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    logger.warning(f"Could not extract video ID from URL: {url}")
    return None


# -----------------------------------------------------------
# STEP 3: Fetch transcript for one video
# -----------------------------------------------------------
def fetch_transcript(video_id: str) -> dict:

    

    result = {
        "video_id":        video_id,
        "status":          None,
        "transcript_text": None,
        "duration_mins":   None,
        "num_segments":    None,
        "raw_segments":    None,
        "error":           None,
    }

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en","en-US", "en-GB"])  # adjust languages if needed
        segments = fetched.to_raw_data()                     # list[dict], like before
        
        full_text = " ".join(seg.get("text", "") for seg in segments).strip()

        if segments:
            last = segments[-1]
            duration_secs = float(last.get("start", 0) or 0) + float(last.get("duration", 0) or 0)
            duration_mins = round(duration_secs / 60, 2)
        else:
            duration_mins = 0.0

        result.update({
            "status":          "success",
            "transcript_text": full_text,
            "duration_mins":   duration_mins,
            "num_segments":    len(segments),
            "raw_segments":    segments,
        })

        logger.info(
            f"✓  {video_id} | {len(segments)} segments | {duration_mins} mins"
        )

    # Known errors — each logged with its own message
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        # Known library exceptions
        result.update({
            "status": "error",
            "error":  f"{type(e).__name__}: {e}",
        })
        logger.error(f"✗  {video_id} | {type(e).__name__} | {e}")
        return result

    # Catch-all — writes full traceback to log file
    except Exception as e:
        msg = str(e)
        result.update({"status": "failed", "error": msg})
        logger.error(
            f"✗  {video_id} | UnexpectedError | {msg}",
            exc_info=True,
        )

    return result


# -----------------------------------------------------------
# STEP 4: Clean transcript text
# -----------------------------------------------------------
def clean_transcript(text: str) -> str:
    """
    Clean raw transcript text:
      - Remove [Music], [Applause], [Laughter] tags
      - Remove (Music), (Applause) tags
      - Collapse multiple spaces/newlines to single space
    """
    if not text:
        return ""

    text = re.sub(r"\[.*?\]", "", text)   # [Music]
    text = re.sub(r"\(.*?\)", "", text)   # (Music)
    text = re.sub(r"\s+", " ", text)      # multiple spaces

    return text.strip()


# -----------------------------------------------------------
# STEP 5: Batch fetch all URLs and save to CSV
# -----------------------------------------------------------
def fetch_all_transcripts(
    urls:       list[str] = None,
    sleep_secs: float = 1.0,
    save:       bool  = True,
) -> pd.DataFrame:
    """
    Fetch transcripts for all URLs and save to CSV.

    Parameters
    ----------
    urls        : list of YouTube URLs (loads from file if None)
    sleep_secs  : delay between requests (avoids rate limiting)
    save        : save results to CSV if True

    Returns
    -------
    pd.DataFrame with one row per URL
    """

    columns = [
        "url",
        "video_id",
        "status",
        "error",
        "transcript_text",
        "transcript_clean",
        "duration_mins",
        "num_segments",
        "raw_segments",
        "word_count",
        "fetched_at",
    ]
    if urls is None:
        urls = load_urls()

    # If urls is [] or otherwise empty, return empty df with correct columns
    if not urls:
        logger.info("Starting transcript fetch for 0 URLs")
        logger.info("─" * 60)
        logger.info("No URLs provided. Returning empty dataframe.")
        df_empty = pd.DataFrame(columns=columns)

        # Optional: still save an empty file (usually not needed)
        if save:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            df_empty.to_csv(TRANSCRIPTS_CSV, index=False)
            logger.info(f"✅ Saved empty transcripts file to: {TRANSCRIPTS_CSV}")

        return df_empty

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total   = len(urls)

    logger.info(f"Starting transcript fetch for {total} URLs")
    logger.info("─" * 60)

    for i, url in enumerate(urls, 1):
        fetched_at = datetime.now().isoformat()
        video_id = extract_video_id(url)

        if not video_id:
            logger.error(f"[{i}/{total}] Skipping — bad URL: {url}")
            records.append({
                "url":              url,
                "video_id":         None,
                "status":           "failed",
                "error":            "Could not extract video ID",
                "transcript_text":  None,
                "transcript_clean": None,
                "duration_mins":    None,
                "num_segments":     None,
                "raw_segments":     None,
                "word_count":       None,
                "fetched_at":  fetched_at,
            })
            continue

        logger.info(f"[{i}/{total}] {video_id}  ({url})")

        result = fetch_transcript(video_id) or {}

        raw_status = str(result.get("status") or "").lower()
        if raw_status == "success":
            status = "success"
        else:
            status = "failed"  # treat anything else ("error", None, etc.) as failed

        transcript_text = result.get("transcript_text")

        clean = ""
        if result["transcript_text"]:
            clean = clean_transcript(result["transcript_text"])

        records.append({
            "url":              url,
            "video_id":         video_id,
            "status":           result["status"],
            "error":            result["error"],
            "transcript_text":  result["transcript_text"],
            "transcript_clean": clean,
            "duration_mins":    result["duration_mins"],
            "num_segments":     result["num_segments"],
            "raw_segments":     str(result["raw_segments"]),
            "word_count":       len(clean.split()) if clean else 0,
            "fetched_at":       datetime.now().isoformat(),
        })

        if i < total:
            time.sleep(sleep_secs)

    df = pd.DataFrame.from_records(records, columns=columns)

    # Summary log
    success = int((df["status"] == "success").sum()) if "status" in df.columns else 0
    failed  = int((df["status"] == "failed").sum()) if "status" in df.columns else 0

    logger.info("─" * 60)
    logger.info(f"Done. ✓ {success} succeeded | ✗ {failed} failed | {total} total")

    if failed > 0:
        logger.warning(f"{failed} failed — check {LOG_FILE}")
        for _, row in df[df["status"] == "failed"].iterrows():
            logger.warning(f"  FAILED: {row['url']} — {row['error']}")

    if save:
        df.to_csv(TRANSCRIPTS_CSV, index=False)
        logger.info(f"✅ Saved to: {TRANSCRIPTS_CSV}")

    return df


# -----------------------------------------------------------
# STEP 6: Print readable summary
# -----------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    """Print a readable summary of the fetch results."""

    total   = len(df)
    success = (df["status"] == "success").sum()
    failed  = (df["status"] == "failed").sum()

    print("\n" + "=" * 60)
    print("📋  TRANSCRIPT FETCH SUMMARY")
    print("=" * 60)
    print(f"  Total URLs  : {total}")
    print(f"  ✅ Success  : {success}")
    print(f"  ❌ Failed   : {failed}")

    if success > 0:
        avg_words = df[df["status"] == "success"]["word_count"].mean()
        avg_mins  = df[df["status"] == "success"]["duration_mins"].mean()
        print(f"\n  Avg word count : {avg_words:,.0f} words")
        print(f"  Avg duration   : {avg_mins:.1f} mins")

    if failed > 0:
        print(f"\n  ❌ Failed URLs:")
        for _, row in df[df["status"] == "failed"].iterrows():
            print(f"     {row['url']}")
            print(f"     Reason: {row['error']}")

    print(f"\n  📄 Log file : {LOG_FILE}")
    print(f"  💾 CSV file : {TRANSCRIPTS_CSV}")
    print("=" * 60)