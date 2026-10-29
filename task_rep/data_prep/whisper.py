from __future__ import annotations

import os
from openai import OpenAI

from pathlib import Path
from typing import List, Optional

def transcribe_audio(audio_path: Path) -> dict:
    
    print(f"    Transcribing with Whisper...")
    granularity = ["segment"]
    response = _transcribe_audio_with_whisper(audio_path, granularity)
    # Parse response
    transcript = []
    content = ""

    if hasattr(response, 'segments'):
        for segment in response.segments:
            text = segment.get('text', '').strip()
            if text:
                transcript.append({
                    "text": text,
                    "start": segment.get('start', 0.0),
                    "end": segment.get('end', 0.0)
                })
                content += text + " "
    elif hasattr(response, 'text'):
        # Fallback if segments not available
        content = response.text
        transcript = [{"text": content, "start": 0.0, "end": 0.0}]

    return {
        "transcript": transcript,
        "content": content.strip(),
        "transcript_source": "whisper"
    }

def _transcribe_audio_with_whisper(audio_path: Path, granularity=["segment"]):
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file
        granularity: Granularity of the transcript

    Returns:
        dict with 'transcript' and 'content' fields.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json",
            timestamp_granularities=granularity,
            prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
        )
        response = response.to_dict()
        return response