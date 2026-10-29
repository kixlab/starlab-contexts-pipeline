"""
Simple video downloader and frame extractor.

Videos are stored centrally and shared across all tasks.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2


@dataclass
class VideoDownloadConfig:
    """Configuration for frame extraction."""
    videos_dir: Path

    fps: float = 1.0  # Extract 1 frame per second
    max_dimension: int = 720  # Resize to max 720p
    format: str = "jpg"
    quality: int = 95  # JPEG quality
    audio_format: str = "m4a"
    audio_quality: int = 128  # Audio quality in kbps


class VideoDownloader:
    """
    Downloads YouTube videos, extracts audio, and extracts frames.

    All videos stored in a central location: static/videos/{video_id}/
    """

    def __init__(self, config: VideoDownloadConfig):
        self.config = config

    def get_video_dir(self, video_id: str) -> Path:
        """Get the directory for a video."""
        return self.config.videos_dir / video_id

    def download_video(self, video_id: str, force: bool = False) -> bool:
        """
        Download a YouTube video and audio separately.

        Args:
            video_id: YouTube video ID
            force: Re-download even if video exists

        Returns:
            True if successful or already exists
        """
        video_dir = self.get_video_dir(video_id)
        video_path = video_dir / f"{video_id}.mp4"
        audio_path = video_dir / f"{video_id}.{self.config.audio_format}"

        video_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://www.youtube.com/watch?v={video_id}"

        video_success = True
        audio_success = True

        # Download video if needed
        if not video_path.exists() or force:
            print(f"  Downloading video...")
            video_success = self._download_video_file(url, video_path, video_id)
        else:
            print(f"  Video already exists")

        # Download audio if needed
        if not audio_path.exists() or force:
            print(f"  Downloading audio...")
            audio_success = self._download_audio_file(url, audio_path, video_id)
        else:
            print(f"  Audio already exists")

        return video_success and audio_success

    def _download_video_file(self, url: str, video_path: Path, video_id: str) -> bool:
        """Download video file only."""
        # Try multiple format selectors with fallbacks
        format_selectors = [
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "best[height<=1080]",
            "best",
        ]

        for i, format_selector in enumerate(format_selectors):
            cmd = [
                "yt-dlp",
                "-f", format_selector,
                "--no-warnings",
                "-o", str(video_path),
            ]

            # Add cookie support
            if os.environ.get("YT_DLP_COOKIES_FROM_BROWSER"):
                cmd.extend(["--cookies-from-browser", os.environ["YT_DLP_COOKIES_FROM_BROWSER"]])
            elif os.path.exists(os.path.expanduser("~/.config/yt-dlp/cookies.txt")):
                cmd.extend(["--cookies", os.path.expanduser("~/.config/yt-dlp/cookies.txt")])

            cmd.append(url)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    return True

                # Try next format selector
                if i < len(format_selectors) - 1:
                    continue
                else:
                    print(f"    Failed to download video {video_id}: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                print(f"    Timeout downloading video {video_id}")
                return False
            except FileNotFoundError:
                print(f"    yt-dlp not found. Install with: pip install yt-dlp")
                return False
            except Exception as e:
                print(f"    Error downloading video {video_id}: {e}")
                return False

        return False

    def _download_audio_file(self, url: str, audio_path: Path, video_id: str) -> bool:
        """Download audio file only."""
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--no-warnings",
            "-x",  # Extract audio only
            "--audio-format", self.config.audio_format,
            "--audio-quality", str(self.config.audio_quality),
            "-o", str(audio_path.with_suffix("")),  # yt-dlp will add the extension
        ]

        # Add cookie support
        if os.environ.get("YT_DLP_COOKIES_FROM_BROWSER"):
            cmd.extend(["--cookies-from-browser", os.environ["YT_DLP_COOKIES_FROM_BROWSER"]])
        elif os.path.exists(os.path.expanduser("~/.config/yt-dlp/cookies.txt")):
            cmd.extend(["--cookies", os.path.expanduser("~/.config/yt-dlp/cookies.txt")])

        cmd.append(url)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return True
            else:
                print(f"    Failed to download audio {video_id}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"    Timeout downloading audio {video_id}")
            return False
        except FileNotFoundError:
            print(f"    yt-dlp not found. Install with: pip install yt-dlp")
            return False
        except Exception as e:
            print(f"    Error downloading audio {video_id}: {e}")
            return False

    def extract_frames(self, video_id: str, force: bool = False) -> List[dict]:
        """
        Extract frames from a video.

        Args:
            video_id: YouTube video ID
            force: Re-extract even if frames exist

        Returns:
            List of frame info dicts: {path, time, frame_number}
        """
        video_dir = self.get_video_dir(video_id)
        video_path = video_dir / f"{video_id}.mp4"
        frames_dir = video_dir / "frames"

        # Check if frames already exist
        if frames_dir.exists() and not force:
            existing_frames = sorted(frames_dir.glob(f"{video_id}_frame_*.{self.config.format}"))
            if existing_frames:
                print(f"Video {video_id} already has {len(existing_frames)} frames, skipping extraction")
                return self._build_frame_info(video_id, existing_frames)

        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return []

        frames_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Default fallback

        frame_interval = int(video_fps / self.config.fps)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps

                # Resize if needed
                height, width = frame.shape[:2]
                if max(height, width) > self.config.max_dimension:
                    if width > height:
                        new_width = self.config.max_dimension
                        new_height = int(height * (self.config.max_dimension / width))
                    else:
                        new_height = self.config.max_dimension
                        new_width = int(width * (self.config.max_dimension / height))
                    frame = cv2.resize(frame, (new_width, new_height))

                # Save frame
                frame_filename = f"{video_id}_frame_{extracted_count:05d}.{self.config.format}"
                frame_path = frames_dir / frame_filename

                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.quality],
                )

                extracted_count += 1

            frame_count += 1

        cap.release()
        print(f"Extracted {extracted_count} frames from {video_id}")

        # Return frame info
        frame_files = sorted(frames_dir.glob(f"{video_id}_frame_*.{self.config.format}"))
        return self._build_frame_info(video_id, frame_files)

    def _build_frame_info(self, video_id: str, frame_paths: List[Path]) -> List[dict]:
        """Build frame info from paths."""
        frames = []
        for i, frame_path in enumerate(frame_paths):
            # Calculate approximate timestamp based on frame number
            timestamp = i / self.config.fps
            frames.append({
                "path": str(frame_path),
                "time": round(timestamp, 2),
                "frame_number": i,
            })
        return frames

    def process_video(self, video_id: str, force_download: bool = False, force_extract: bool = False) -> List[dict]:
        """
        Download video and extract frames.

        Args:
            video_id: YouTube video ID
            force_download: Re-download even if exists
            force_extract: Re-extract frames even if exist

        Returns:
            List of frame info dicts
        """
        # Download
        success = self.download_video(video_id, force=force_download)
        if not success:
            return []

        # Extract frames
        return self.extract_frames(video_id, force=force_extract)

    def get_frames(self, video_id: str) -> List[dict]:
        """
        Get frame information for a video (without downloading/extracting).

        Returns:
            List of frame info dicts, or empty list if frames don't exist
        """
        frames_dir = self.get_video_dir(video_id) / "frames"
        if not frames_dir.exists():
            return []

        frame_files = sorted(frames_dir.glob(f"{video_id}_frame_*.{self.config.format}"))
        return self._build_frame_info(video_id, frame_files)
    
    def get_audio(self, video_id: str) -> Path:
        """Get audio path."""
        video_dir = self.get_video_dir(video_id)
        audio_path = video_dir / f"{video_id}.{self.config.audio_format}"
        return audio_path
    
    def has_audio(self, video_id: str) -> bool:
        """Check if audio is extracted."""
        audio_path = self.get_video_dir(video_id) / f"{video_id}.{self.config.audio_format}"
        return audio_path.exists()

    def has_video(self, video_id: str) -> bool:
        """Check if video is downloaded."""
        video_path = self.get_video_dir(video_id) / f"{video_id}.mp4"
        return video_path.exists()

    def has_frames(self, video_id: str) -> bool:
        """Check if frames are extracted."""
        frames_dir = self.get_video_dir(video_id) / "frames"
        return frames_dir.exists() and any(frames_dir.glob(f"{video_id}_frame_*"))

    def get_transcript_path(self, video_id: str, source: str = "openai") -> Path:
        """
        Get transcript file path for a specific source.

        Args:
            video_id: Video ID
            source: Transcript source ("openai", "youtube", or "none")
        """
        return self.get_video_dir(video_id) / f"transcript_{source}.json"

    def has_transcript(self, video_id: str, source: str = "openai") -> bool:
        """
        Check if transcript exists for a specific source.

        Args:
            video_id: Video ID
            source: Transcript source ("openai" or "youtube")
        """
        return self.get_transcript_path(video_id, source).exists()

    def get_transcript(self, video_id: str, source: str = "openai") -> Optional[dict]:
        """
        Load transcript from file if it exists.

        Args:
            video_id: Video ID
            source: Transcript source ("openai" or "youtube")

        Returns:
            Transcript dict with 'transcript', 'content', and 'transcript_source' fields,
            or None if no transcript exists.
        """
        transcript_path = self.get_transcript_path(video_id, source)
        if not transcript_path.exists():
            return None

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"    Failed to load transcript: {e}")
            return None

    def save_transcript(self, video_id: str, transcript_data: dict) -> bool:
        """
        Save transcript to file based on its source.

        Args:
            video_id: Video ID
            transcript_data: Dict with 'transcript', 'content', and 'transcript_source' fields

        Returns:
            True if successful, False otherwise
        """
        source = transcript_data.get("transcript_source", "openai")
        transcript_path = self.get_transcript_path(video_id, source)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"    Failed to save transcript: {e}")
            return False
