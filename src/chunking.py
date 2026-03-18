# src/chunking.py
# -----------------------------------------------------------
# Token-based chunking with precise timestamp mapping
# -----------------------------------------------------------

import ast
from typing import List, Dict
from src.timestamp_utils import format_timestamp, format_duration, calculate_segment_end_time
from src.logging_utils import get_logger

logger = get_logger("chunking")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_transcript_with_timestamps(
    raw_segments: list,
    episode_metadata: dict,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[Dict]:
    
    if not raw_segments:
        logger.warning(f"Episode {episode_metadata.get('episode_id')}: No segments found")
        return []
    
    chunks = []
    current_chunk_segments = []
    current_tokens = 0
    overlap_segments = []  # Segments to carry over for overlap
    
    for seg_idx, segment in enumerate(raw_segments):
        text = segment.get('text', '').strip()
        if not text:
            continue
        
        tokens_in_segment = estimate_tokens(text)
        
        # Check if adding this segment exceeds chunk limit
        if current_tokens + tokens_in_segment > chunk_size_tokens and current_chunk_segments:
            # Save current chunk
            chunk = _create_chunk_from_segments(
                segments=current_chunk_segments,
                episode_metadata=episode_metadata,
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
            
            logger.debug(
                f"Chunk {len(chunks)}: {chunk['start_time_display']} - {chunk['end_time_display']} "
                f"({chunk['token_count']} tokens, {len(current_chunk_segments)} segments)"
            )
            
            # Prepare overlap: keep last few segments
            overlap_segments = _get_overlap_segments(
                current_chunk_segments, 
                overlap_tokens
            )
            
            # Start new chunk with overlap segments
            current_chunk_segments = overlap_segments.copy()
            current_tokens = sum(estimate_tokens(s['text']) for s in overlap_segments)
        
        # Add current segment to chunk
        current_chunk_segments.append(segment)
        current_tokens += tokens_in_segment
    
    # Don't forget the last chunk
    if current_chunk_segments:
        chunk = _create_chunk_from_segments(
            segments=current_chunk_segments,
            episode_metadata=episode_metadata,
            chunk_index=len(chunks),
        )
        chunks.append(chunk)
        
        logger.debug(
            f"Chunk {len(chunks)} (final): {chunk['start_time_display']} - {chunk['end_time_display']} "
            f"({chunk['token_count']} tokens)"
        )
    
    # Update total_chunks for all chunks
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)
    
    logger.info(
        f"Episode {episode_metadata.get('episode_id')}: "
        f"Created {len(chunks)} chunks from {len(raw_segments)} segments"
    )
    
    return chunks


def _create_chunk_from_segments(
    segments: list,
    episode_metadata: dict,
    chunk_index: int,
) -> dict:
  
    # Combine segment texts
    chunk_text = ' '.join(seg['text'].strip() for seg in segments if seg.get('text'))
    
    # Calculate timestamps
    first_segment = segments[0]
    last_segment = segments[-1]
    
    start_time_seconds = first_segment['start']
    end_time_seconds = calculate_segment_end_time(last_segment)
    duration_seconds = end_time_seconds - start_time_seconds
    
    # Format timestamps for display
    start_time_display = format_timestamp(start_time_seconds)
    end_time_display = format_timestamp(end_time_seconds)
    duration_display = format_duration(duration_seconds)
    
    # Calculate token and word counts
    token_count = estimate_tokens(chunk_text)
    word_count = len(chunk_text.split())
    
    # Create chunk ID
    chunk_id = f"{episode_metadata['episode_id']}_c{chunk_index:03d}"
    
    return {
        # Identity
        'chunk_id': chunk_id,
        'episode_id': episode_metadata['episode_id'],
        
        # Text content
        'text': chunk_text,
        'token_count': token_count,
        'word_count': word_count,
        
        # Episode info
        'episode_title': episode_metadata['youtube_title'],
        'youtube_channel': episode_metadata['youtube_channel'],
        'url': episode_metadata['url'],
        'video_id': episode_metadata.get('video_id', ''),
        
        # Timestamp data
        'start_time_seconds': int(start_time_seconds),
        'end_time_seconds': int(end_time_seconds),
        'duration_seconds': int(duration_seconds),
        'start_time_display': start_time_display,
        'end_time_display': end_time_display,
        'duration_display': duration_display,
        
        # Chunk positioning
        'chunk_index': chunk_index,
        'total_chunks': None,  # Set after all chunks created
        
        # Segment mapping (for reconstruction if needed)
        'num_segments': len(segments),
    }


def _get_overlap_segments(segments: list, overlap_tokens: int) -> list:

    if not segments or overlap_tokens <= 0:
        return []
    
    overlap_segs = []
    accumulated_tokens = 0
    
    # Walk backwards through segments
    for segment in reversed(segments):
        text = segment.get('text', '').strip()
        if not text:
            continue
        
        tokens = estimate_tokens(text)
        
        # Stop if we've accumulated enough overlap
        if accumulated_tokens + tokens > overlap_tokens and overlap_segs:
            break
        
        overlap_segs.insert(0, segment)  # Add to front (preserve order)
        accumulated_tokens += tokens
    
    return overlap_segs


def parse_raw_segments(raw_segments_str: str) -> list:
   
    try:
        return ast.literal_eval(raw_segments_str)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse raw_segments: {e}")
        return []