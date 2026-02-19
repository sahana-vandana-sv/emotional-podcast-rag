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

from src.config import BASE_DIR, PROCESSED_DIR


# -----------------------------------------------------------
# PATHS - update in config 
# -----------------------------------------------------------
URLS_FILE = BASE_DIR / "data" / "podcast_urls.csv"
OUTPUT_CSV = PROCESSED_DIR / "transcripts_df.csv"
LOG_DIR    = BASE_DIR / "logs"
LOG_FILE   = LOG_DIR / "transcript_errors.log"


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
def load_urls(filepath: Path = URLS_FILE) -> list[str]:
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
    """
    Fetch the transcript for a single YouTube video.

    Returns
    -------
    dict with keys:
        video_id, status, transcript_text, duration_mins,
        num_segments, raw_segments, error (if failed)
    """
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
        segments = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "en-US", "en-GB"],
        )

        full_text = " ".join(seg["text"] for seg in segments)

        if segments:
            last = segments[-1]
            duration_secs = last.get("start", 0) + last.get("duration", 0)
            duration_mins = round(duration_secs / 60, 2)
        else:
            duration_mins = 0

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
    except TranscriptsDisabled:
        msg = "Transcripts are disabled for this video"
        result.update({"status": "failed", "error": msg})
        logger.warning(f"✗  {video_id} | TranscriptsDisabled | {msg}")

    except NoTranscriptFound:
        msg = "No English transcript found"
        result.update({"status": "failed", "error": msg})
        logger.warning(f"✗  {video_id} | NoTranscriptFound | {msg}")

    except VideoUnavailable:
        msg = "Video is unavailable or private"
        result.update({"status": "failed", "error": msg})
        logger.error(f"✗  {video_id} | VideoUnavailable | {msg}")

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
    if urls is None:
        urls = load_urls()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total   = len(urls)

    logger.info(f"Starting transcript fetch for {total} URLs")
    logger.info("─" * 60)

    for i, url in enumerate(urls, 1):

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
                "fetched_at":       datetime.now().isoformat(),
            })
            continue

        logger.info(f"[{i}/{total}] {video_id}  ({url})")

        result = fetch_transcript(video_id)

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

    df = pd.DataFrame(records)

    # Summary log
    success = (df["status"] == "success").sum()
    failed  = (df["status"] == "failed").sum()

    logger.info("─" * 60)
    logger.info(f"Done. ✓ {success} succeeded | ✗ {failed} failed | {total} total")

    if failed > 0:
        logger.warning(f"{failed} failed — check {LOG_FILE}")
        for _, row in df[df["status"] == "failed"].iterrows():
            logger.warning(f"  FAILED: {row['url']} — {row['error']}")

    if save:
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"✅ Saved to: {OUTPUT_CSV}")

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
    print(f"  💾 CSV file : {OUTPUT_CSV}")
    print("=" * 60)