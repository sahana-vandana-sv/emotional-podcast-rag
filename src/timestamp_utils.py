
def format_timestamp(seconds:float)->str:
    # Convert seconds to human-readable timestamp.
    seconds = int(seconds)
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"
    
def parse_timestamp(timestamp_str:str)->int:
    #Convert timestamp string to seconds.
    parts = timestamp_str.split(':')
    
    if len(parts) == 2:  # M:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # H:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    
def format_duration(seconds: float) -> str:
    # Convert duration in seconds to readable format.

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if nothing else
        parts.append(f"{secs}s")
    
    return ' '.join(parts)

def calculate_segment_end_time(segment: dict) -> float:
    # Calculate end time of a segment.

    return segment['start'] + segment.get('duration', 0)

