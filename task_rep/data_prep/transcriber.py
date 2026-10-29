from __future__ import annotations

import os
import subprocess
from openai import OpenAI
from pathlib import Path
from typing import Optional

# Whisper API has a 25 MB file size limit, use 24 MB to be safe
MAX_FILE_SIZE_MB = 24
CHUNK_DURATION_SECONDS = 600  # 10 minutes per chunk
MIN_CHUNK_DURATION_SECONDS = 60  # Minimum chunk duration

def transcribe_audio(audio_path: Path) -> dict:

    print(f"    Transcribing with OpenAI...")

    # Check file size
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"    Audio file is {file_size_mb:.1f} MB, splitting into chunks...")
        return _transcribe_large_audio(audio_path)
    else:
        print(f"    Audio file is {file_size_mb:.1f} MB, transcribing directly...")
        response = _transcribe_audio_with_openai(audio_path, prompt=None)
        return _parse_openai_response(response, time_offset=0.0)

def _transcribe_large_audio(audio_path: Path) -> dict:
    """
    Transcribe large audio files by splitting them into chunks.
    Uses dynamic overlap based on last segment duration to avoid cutting sentences.
    """
    # Get audio duration using ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
    except Exception as e:
        print(f"    Failed to get audio duration: {e}")
        return {"transcript": [], "content": "", "transcript_source": "whisper"}

    print(f"    Audio duration: {duration:.1f} seconds")

    # Create temp directory for chunks
    temp_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    temp_dir.mkdir(exist_ok=True)

    all_transcripts = []
    next_start_time = 0.0
    chunk_num = 0

    try:
        while next_start_time < duration:
            chunk_num += 1
            chunk_path = temp_dir / f"chunk_{chunk_num:03d}.m4a"

            # Calculate chunk duration
            remaining = duration - next_start_time
            chunk_duration = min(CHUNK_DURATION_SECONDS, remaining)

            # If what remains after this chunk would be too short, extend this chunk to include it
            if remaining > chunk_duration and (remaining - chunk_duration) < MIN_CHUNK_DURATION_SECONDS:
                chunk_duration = remaining
                print(f"    Extending chunk to include remaining audio ({remaining:.1f}s total)")

            print(f"    Processing chunk {chunk_num} (start: {next_start_time:.1f}s, duration: {chunk_duration:.1f}s)...")

            # Extract chunk using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(next_start_time),
                "-t", str(chunk_duration),
                "-c", "copy",
                "-v", "quiet",
                str(chunk_path)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode != 0:
                print(f"    Failed to extract chunk {chunk_num}")
                break

            # Transcribe chunk with context from previous segment
            try:
                # Get previous context of up to 400 characters
                previous_context = ""
                if len(all_transcripts) > 0:
                    for segment in all_transcripts[::-1]:
                        if segment["end"] <= next_start_time and len(previous_context) < 400:
                            previous_context = segment["text"] + " " + previous_context

                # Trim context to 400 characters if it's too long
                if len(previous_context) > 400:
                    previous_context = previous_context[-400:]
                
                if len(previous_context.strip()) == 0:
                    previous_context = None

                response = _transcribe_audio_with_openai(chunk_path, prompt=previous_context)
                chunk_data = _parse_openai_response(response, time_offset=next_start_time)

                while len(all_transcripts) > 0 and all_transcripts[-1]["end"] > next_start_time: 
                    removed_segment = all_transcripts.pop()
                    print(f"    Removed overlapping segment: '{removed_segment['text'][:50]}...'")

                # Add all segments from this chunk
                all_transcripts.extend(chunk_data["transcript"])

                # Prepare for next iteration:
                # - If we have multiple segments, use the last segment's start time as next chunk start
                #   and its text as context
                # - If only one segment, move to end of chunk
                if len(chunk_data["transcript"]) > 1 and remaining > chunk_duration: 
                    # If we have multiple segments and there's remaining audio, remove the last segment
                    last_segment = chunk_data["transcript"][-1]
                    next_start_time = last_segment["start"]

                    print(f"    Next chunk will start at {next_start_time:.1f}s")
                else:
                    # Only one segment, move to end of current chunk
                    next_start_time += chunk_duration
                    print(f"    Single segment, moving to {next_start_time:.1f}s")

                print(f"    ✓ Chunk {chunk_num}: {len(chunk_data['transcript'])} segments")

            except Exception as e:
                print(f"    Failed to transcribe chunk {chunk_num}: {e}")
                break
            finally:
                # Clean up chunk file
                if chunk_path.exists():
                    chunk_path.unlink()

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except:
                pass

    return all_transcripts

def _parse_openai_response(response: dict, time_offset: float = 0.0) -> dict:
    """
    Parse Whisper API response and extract transcript with time offset.

    Args:
        response: Whisper API response as dict
        time_offset: Time offset to add to all timestamps (for chunked audio)
    """
    transcript = []

    if 'segments' in response:
        for segment in response['segments']:
            text = segment.get('text', '').strip()
            if text:
                transcript.append({
                    "text": text,
                    "start": segment.get('start', 0.0) + time_offset,
                    "end": segment.get('end', 0.0) + time_offset
                })
    elif 'text' in response:
        # Fallback if segments not available
        transcript = [{"text": response['text'], "start": time_offset, "end": time_offset}]

    return transcript

def _transcribe_audio_with_openai(audio_path: Path, prompt: Optional[str] = None):
    """
    Transcribe audio using OpenAI Whisper API (whisper-1).

    Args:
        audio_path: Path to the audio file
        prompt: Optional context from previous segment for better continuity

    Returns:
        dict with transcript segments and content.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # Use context prompt if provided, otherwise use default
    if prompt is None:
        prompt = "Umm, let me think like, hmm, okay, here's what I'm, like, thinking."

    with open(audio_path, "rb") as audio:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            prompt=prompt
        )
        response = response.to_dict()
        return response