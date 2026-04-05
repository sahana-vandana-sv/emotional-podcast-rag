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
import os
import re
import time
import random
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
from youtube_transcript_api._errors import IpBlocked, RequestBlocked

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

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    logger.warning(
        "google-api-python-client not installed. "
        "Metadata fetching will be limited. "
        "Install with: pip install google-api-python-client"
    )


YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Set in .env

if YOUTUBE_API_KEY and YOUTUBE_API_AVAILABLE:
    youtube_api = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    logger.info("✓ YouTube Data API v3 initialized")
else:
    youtube_api = None
    if YOUTUBE_API_AVAILABLE:
        logger.warning(
            "YOUTUBE_API_KEY not set. Add to .env file. "
            "Get key from: https://console.cloud.google.com"
        )


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

# STEP 3A: Fetch metadata using YouTube Data API v3

def _parse_iso_duration(iso_duration: str) -> float:
    """Convert ISO 8601 duration string (e.g. PT1H23M45S) to total seconds."""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return (int(hours or 0) * 3600 +
            int(minutes or 0) * 60 +
            int(seconds or 0))


def fetch_metadata_youtube_api(video_id: str) -> dict:
    result = {
        "youtube_title":   None,
        "youtube_channel": None,
        "duration_mins":   None,
        "error":           None,
    }
    
    if not youtube_api:
        result["error"] = "YouTube API not configured"
        return result

    try:
        request = youtube_api.videos().list(
            part="snippet,contentDetails",
            id=video_id
        )
        response = request.execute()
        
        if not response.get('items'):
            result["error"] = "Video not found"
            logger.warning(f"Video not found via API: {video_id}")
            return result
        
        item = response['items'][0]
        snippet = item.get('snippet', {})
        content_details = item.get('contentDetails', {})
        
        # Extract metadata
        result["youtube_title"] = snippet.get('title', 'Unknown Title')
        result["youtube_channel"] = snippet.get('channelTitle', 'Unknown Channel')
        
        logger.debug(
            f"✓ API metadata: {result['youtube_title'][:40]} | "
            f"{result['youtube_channel']} | {result['duration_mins']} mins"
        )
        
    except HttpError as e:
        result["error"] = f"API error: {e.resp.status}"
        logger.error(f"YouTube API error for {video_id}: {e}")
        
    except Exception as e:
        result["error"] = f"Unexpected error: {type(e).__name__}"
        logger.error(f"Error fetching metadata for {video_id}: {e}")

    return result

def parse_iso8601_duration(duration: str) -> float:

    """
    Convert ISO 8601 duration to minutes. 
    Examples:
        "PT1H2M10S" → 62.17 minutes
        "PT5M30S"   → 5.5 minutes
        "PT45S"     → 0.75 minutes
    """
    import re
    match = re.match(
        r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0.0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    total_minutes = hours * 60 + minutes + seconds / 60

    return round(total_minutes, 2)

# -----------------------------------------------------------
# STEP 3: Fetch transcript for one video
# -----------------------------------------------------------
def fetch_transcript(video_id: str, max_retries: int = 5,base_wait: float = 10.0,use_proxy: bool = False)) -> dict:

    result = {
        "video_id":        video_id,
        "youtube_title":None,
        "youtube_channel":None,
        "status":          None,
        "transcript_text": None,
        "duration_mins":   None,
        "num_segments":    None,
        "raw_segments":    None,
        "error":           None,
    }

    # First: Try to get metadata from YouTube API
   
    if youtube_api:
        metadata = fetch_metadata_youtube_api(video_id)
        result["youtube_title"] = metadata.get("youtube_title")
        result["youtube_channel"] = metadata.get("youtube_channel")

        if metadata.get("error") and not result["youtube_title"]:
            logger.warning(f"Could not fetch metadata via API: {metadata['error']}")


    ytt_api = YouTubeTranscriptApi()

    for attempt in range(1, max_retries + 1):
        try:
            # Fetch transcript with metadata
            fetched = ytt_api.fetch(video_id, languages=["en","en-US", "en-GB"]) 
            #Extract segments
            segments = fetched.to_raw_data() or []              
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
                "error": None,
            })

            logger.info(
                f"✓  {video_id} | {len(segments)} segments | {duration_mins} mins"
            )
            return result

        # Known errors — each logged with its own message
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            # Known library exceptions
            result.update({
                "status": "unavailable",
                "error":  f"{type(e).__name__}: {e}",
            })
            logger.error(f"✗  {video_id} | {type(e).__name__} | {e}")
            return result
                # Explicit IP/request blocks (usually retryable only with long cooldown; often best to stop)
        except (IpBlocked, RequestBlocked) as e:
            result.update({
                "status": "ip_blocked",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            logger.error(f"⛔  {video_id} | {type(e).__name__} | {e}")
            return result

        # Other errors: retry with backoff
        except Exception as e:
            msg = str(e) or ""
            is_rate_limit_like = any(
                s in msg.lower()
                for s in ["too many requests", "blocking requests", "429", "rate limit", "temporarily unavailable"]
            )

            # If we can retry, wait with exponential backoff + small jitter
            if attempt < max_retries and is_rate_limit_like:
                wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 2)

                logger.warning(
                    f"⚠️  {video_id} | Temporary block/rate-limit "
                    f"(attempt {attempt}/{max_retries}). Waiting {wait:.1f}s..."

                )
                time.sleep(wait)
                continue

           # Final failure or not retryable
            result.update({
                "status": "failed",
                "error": f"{type(e).__name__}: {msg[:200]}",
            })
            
            logger.error(
                    f"✗  {video_id} | {type(e).__name__} | "
                f"{'rate-limit-like' if is_rate_limit_like else 'error'} after {attempt} attempts",
                exc_info=True,
                )
            
            return result
        # Should rarely reach here
    if result["status"] is None:
        result["status"] = "failed"
        result["error"] = "Unknown failure after retries"


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
    keep_failed: bool = False, 
) -> pd.DataFrame:

    columns = [
        "url",
        "video_id",
        "youtube_title",      
        "youtube_channel",
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
                "youtube_title":    None,
                "youtube_channel":  None,
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
           "youtube_title":    result.get("youtube_title"),
            "youtube_channel":  result.get("youtube_channel"),
            "status":           result.get("status"),
            "error":            result.get("error"),
            "transcript_text":  result.get("transcript_text"),
            "transcript_clean": clean,
            "duration_mins":    result.get("duration_mins"),
            "num_segments":     result.get("num_segments"),
            "raw_segments":     result.get("raw_segments"),
            "word_count":       len(clean.split()) if clean else 0,
            "fetched_at":       datetime.now().isoformat(),
        })
        
        # Rate limiting
        if i < total:
            time.sleep(sleep_secs)

    df = pd.DataFrame(records, columns=columns)

    # Summary log
    success = int((df["status"] == "success").sum()) if "status" in df.columns else 0
    failed  = int((df["status"] == "failed").sum()) if "status" in df.columns else 0

    logger.info("─" * 60)
    logger.info(f"Done. ✓ {success} succeeded | ✗ {failed} failed | {total} total")

    if failed > 0:
        logger.warning(f"{failed} failed — check {LOG_FILE}")
        for _, row in df[df["status"] == "failed"].iterrows():
            logger.warning(f"  FAILED: {row['url']} — {row['error']}")
        if not keep_failed:
            logger.info(f"\nRemoving {failed} failed rows from CSV...")

    # Remove failed rows (unless keep_failed=True)
    if not keep_failed:
        df_clean = df[df["status"] == "success"].copy()
        removed_count = len(df) - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"✂️  Removed {removed_count} failed rows")
            logger.info(f"   Keeping {len(df_clean)} successful rows")
        
        df = df_clean      
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