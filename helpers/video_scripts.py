def get_transcript_segment(transcript, start, end, include_intersecting=False):
    segment_str = ""
    for segment in transcript:
        if include_intersecting:
            if max(segment["start"], start) <= min(segment["end"], end):
                segment_str += segment["text"] + " "
        elif segment["start"] >= start and segment["end"] <= end:
            segment_str += segment["text"] + " "
    return segment_str.strip()