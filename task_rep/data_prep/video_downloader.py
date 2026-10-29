"""
Simple video downloader and frame extractor.

Videos are stored centrally and shared across all tasks.
"""

from __future__ import annotations

import json
import os
import subprocess
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

from .transcriber import transcribe_audio

def _add_cookie_support(cmd: List[str]) -> List[str]:
    cmd.extend(["--js-runtimes", "node"])
    cookies_file = os.environ.get("YT_DLP_COOKIES_FILE")
    browser = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER")
    if cookies_file:
        cmd.extend(["--cookies", str(Path(cookies_file).expanduser())])
    elif browser:
        cmd.extend(["--cookies-from-browser", browser])
    return cmd

@dataclass
class VideoDownloader:
    """
    Downloads YouTube videos, extracts audio, and extracts frames.

    All videos stored in a central location: static/videos/{video_id}/
    """

    videos_dir: Path
    video_format: str = "mp4"
    frame_max_dimension: int = 720  # Resize to max 720p
    frame_format: str = "jpg"
    frame_quality: int = 95  # JPEG quality
    audio_format: str = "m4a"
    audio_quality: int = 128  # Audio quality in kbps

    def get_video_url(self, video_id: str) -> str:
        """Get the URL of a video."""
        return f"https://www.youtube.com/watch?v={video_id}"

    def get_video_dir(self, video_id: str) -> Path:
        """Get the directory for a video."""
        return self.videos_dir / video_id

    def get_metadata_path(self, video_id: str) -> Path:
        """Get the path for a metadata."""
        return self.get_video_dir(video_id) / f"{video_id}_metadata.json"

    def get_video_path(self, video_id: str) -> Path:
        """Get the path for a video."""
        return self.get_video_dir(video_id) / f"{video_id}.{self.video_format}"

    def get_audio_path(self, video_id: str) -> Path:
        """Get the path for a audio."""
        return self.get_video_dir(video_id) / f"{video_id}.{self.audio_format}"

    def get_youtube_transcript_path(self, video_id: str) -> Path:
        """Get the path for a transcript."""
        return self.get_video_dir(video_id) / f"{video_id}.youtube.json"
    
    def get_openai_transcript_path(self, video_id: str) -> Path:
        """Get the path for a transcript."""
        return self.get_video_dir(video_id) / f"{video_id}.openai.json"

    def get_frames_dir(self, video_id: str) -> Path:
        """Get the directory for frames."""
        return self.get_video_dir(video_id) / "frames"

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
        video_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self.get_metadata_path(video_id)
        video_path = self.get_video_path(video_id)
        audio_path = self.get_audio_path(video_id)
        youtube_transcript_path = self.get_youtube_transcript_path(video_id)
        openai_transcript_path = self.get_openai_transcript_path(video_id)

        metadata_success = True 
        video_success = True
        audio_success = True
        youtube_transcript_success = True
        openai_transcript_success = True

        url = self.get_video_url(video_id)

        # Fetch metadata if needed
        if not metadata_path.exists() or force:
            print(f"  Fetching metadata...")
            metadata_success = self._fetch_metadata(video_id, url, metadata_path)
        else:
            print(f"  Metadata already exists")
            metadata_success = True

        # Download video if needed
        if not video_path.exists() or force:
            print(f"  Downloading video...")
            video_success = self._download_video_file(url, video_path)
        else:
            print(f"  Video already exists")

        # Download audio if needed
        if not audio_path.exists() or force:
            print(f"  Downloading audio...")
            audio_success = self._download_audio_file(url, audio_path)
        else:
            print(f"  Audio already exists")

        if not youtube_transcript_path.exists() or force:
            print(f"  Fetching YouTube transcript...")
            youtube_transcript_success = self._fetch_youtube_transcript(url, youtube_transcript_path)
        else:
            print(f"  YouTube transcript already exists")
            youtube_transcript_success = True

        if audio_success:
            if not openai_transcript_path.exists() or force:
                print(f"  Fetching OpenAI transcript...")
                openai_transcript_success = self._generate_openai_transcript(audio_path, openai_transcript_path)
            else:
                print(f"  OpenAI transcript already exists")
                openai_transcript_success = True
        else:
            openai_transcript_success = False

        transcript_success = youtube_transcript_success or openai_transcript_success

        return metadata_success and video_success and audio_success and transcript_success

    def _fetch_metadata(self, video_id: str, url: str, metadata_path: Path) -> bool:
        """Fetch metadata, trying YouTube Data API v3 first, then yt-dlp as fallback."""
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if api_key:
            try:
                data = self._fetch_metadata_via_youtube_api(video_id, api_key)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                print(f"    YouTube Data API failed, falling back to yt-dlp: {e}")
        else:
            print("    YOUTUBE_API_KEY not set, falling back to yt-dlp")

        return self._fetch_metadata_via_ytdlp(url, metadata_path)

    def _fetch_metadata_via_youtube_api(self, video_id: str, api_key: str) -> dict:
        """
        Fetch metadata from the YouTube Data API v3.

        Raises on any failure so the caller can fall back to yt-dlp.
        Only requests the parts needed for the summary fields.
        """
        import urllib.request
        import urllib.parse
        import re

        params = urllib.parse.urlencode({
            "id": video_id,
            "part": "snippet,contentDetails,statistics",
            "key": api_key,
        })
        api_url = f"https://www.googleapis.com/youtube/v3/videos?{params}"
        with urllib.request.urlopen(api_url, timeout=30) as resp:
            body = json.loads(resp.read())

        items = body.get("items", [])
        if not items:
            raise ValueError(f"Video not found or API quota exceeded for {video_id}")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})

        # Parse ISO 8601 duration (e.g. "PT1H3M42S") into seconds
        duration_str = details.get("duration", "PT0S")
        m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
        h, mins, s = (int(x or 0) for x in m.groups()) if m else (0, 0, 0)
        duration_secs = h * 3600 + mins * 60 + s

        published_at = snippet.get("publishedAt", "")  # e.g. "2009-10-25T06:57:33Z"

        def _int_or_none(v):
            return int(v) if v is not None else None

        return {
            "id": video_id,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "creator": snippet.get("channelTitle"),
            "channel_id": snippet.get("channelId"),
            "tags": snippet.get("tags", []),
            "duration": duration_secs,
            "upload_date": published_at[:10].replace("-", ""),  # "YYYYMMDD"
            "release_year": int(published_at[:4]) if published_at else None,
            "view_count": _int_or_none(stats.get("viewCount")),
            "like_count": _int_or_none(stats.get("likeCount")),
            "dislike_count": None,  # Removed by YouTube in 2021
            "comment_count": _int_or_none(stats.get("commentCount")),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
            "_source": "youtube_api_v3",
        }

    def _fetch_metadata_via_ytdlp(self, url: str, metadata_path: Path) -> bool:
        """Fetch metadata via yt-dlp (fallback when YouTube API is unavailable)."""
        try:
            import yt_dlp
        except ImportError:
            print("    yt-dlp not found. Install with: pip install yt-dlp")
            return False

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "no_check_formats": True,
            "retries": 3,
            "socket_timeout": 30,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                data = ydl.extract_info(url, download=False)
                data = ydl.sanitize_info(data)

            if not data.get("creator"):
                data["creator"] = data.get("uploader") or data.get("channel")
            if not data.get("release_year"):
                upload_date = data.get("upload_date") or data.get("release_date") or ""
                if len(upload_date) >= 4:
                    data["release_year"] = int(upload_date[:4])

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except yt_dlp.utils.DownloadError as e:
            print(f"    Failed to get video metadata `{url}`: {e}")
            return False
        except Exception as e:
            print(f"    Error getting video metadata `{url}`: {e}")
            return False

    def _download_video_file(self, url: str, video_path: Path) -> bool:
        """Download video file only."""
        # Try multiple format selectors with fallbacks
        format_selectors = [
            "bestvideo[height<=720][ext=mp4][vcodec!=av01]+bestaudio[ext=m4a]/best[height<=720][ext=mp4][vcodec!=av01]/best[height<=720][vcodec!=av01]",
            "bestvideo[height<=720][vcodec!=av01]+bestaudio/best[height<=720][vcodec!=av01]",
            "best[height<=1080][vcodec!=av01]",
            "best[vcodec!=av01]",
            "best",  # last resort: no codec restriction
        ]

        for i, format_selector in enumerate(format_selectors):
            cmd = [
                "yt-dlp",
                "-f", format_selector,
                "--no-warnings",
                "--retries", "3",
                "--sleep-requests", "1",
                "-o", str(video_path),
            ]

            cmd = _add_cookie_support(cmd)
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
                    print(f"    Failed to download video `{url}`: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                print(f"    Timeout downloading video `{url}`")
                return False
            except FileNotFoundError:
                print(f"    yt-dlp not found. Install with: pip install yt-dlp")
                return False
            except Exception as e:
                print(f"    Error downloading video `{url}`: {e}")
                return False

        return False

    def _download_audio_file(self, url: str, audio_path: Path) -> bool:
        """Download audio file only."""
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--no-warnings",
            "--retries", "3",
            "--sleep-requests", "1",
            "-x",  # Extract audio only
            "--audio-format", self.audio_format,
            "--audio-quality", str(self.audio_quality),
            "-o", str(audio_path.with_suffix("")),  # yt-dlp will add the extension
        ]

        cmd = _add_cookie_support(cmd)
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
                print(f"    Failed to download audio `{url}`: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"    Timeout downloading audio `{url}`")
            return False
        except FileNotFoundError:
            print(f"    yt-dlp not found. Install with: pip install yt-dlp")
            return False
        except Exception as e:
            print(f"    Error downloading audio `{url}`: {e}")
            return False

    def _fetch_youtube_transcript(self, url: str, transcript_path: Path) -> bool:
        """
        Fetch original YouTube transcript using yt-dlp.

        Returns True if successful, False otherwise.
        """
        return False ### Skipped for now due to YouTube throttling
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "-o", str(transcript_path.with_suffix(".json3")),
        ]

        cmd = _add_cookie_support(cmd)
        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"    Failed to fetch YouTube transcript `{url}`: {result.stderr}")
                return False

            # Try to load the subtitle file
            sub_file = transcript_path
            if sub_file.exists():
                with open(sub_file, "r", encoding="utf-8") as f:
                    sub_data = json.load(f)

                # Parse transcript
                transcript = []
                for event in sub_data.get("events", []):
                    if "segs" in event:
                        text = "".join(seg.get("utf8", "") for seg in event["segs"])
                        start = event.get("tStartMs", 0) / 1000.0
                        duration = event.get("dDurationMs", 0) / 1000.0

                        if text.strip():
                            transcript.append({
                                "text": text.strip(),
                                "start": start,
                                "end": start + duration
                            })

                # Clean up subtitle file
                sub_file.unlink()

                transcript_data = {
                    "transcript": transcript,
                    "transcript_source": "youtube"
                }

                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                
                return True
        except subprocess.TimeoutExpired:
            print(f"    Timeout fetching YouTube transcript `{url}`")
            return False
        except FileNotFoundError:
            print(f"    yt-dlp not found. Install with: pip install yt-dlp")
            return False
        except Exception as e:
            print(f"    Error fetching YouTube transcript `{url}`: {e}")
            return False

    def _generate_openai_transcript(self, audio_path: Path, transcript_path: Path) -> bool:
        """
        Generate OpenAI transcript using OpenAI API.

        Returns True if successful, False otherwise.
        """

        if not audio_path.exists():
            print(f"    Audio file `{audio_path}` not found")
            return False

        try:   
            transcript = transcribe_audio(audio_path)
            transcript_data = {
                "transcript": transcript,
                "transcript_source": "openai"
            }
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"    Error generating OpenAI transcript `{audio_path}`: {e}")
            return False
        
    
    def extract_frames(self, video_id: str, fps: float) -> List[dict]:
        """
        Extract frames from a video.

        Args:
            video_id: YouTube video ID

        Returns:
            List of frame info dicts: {path, time, frame_number}
        """
        video_path = self.get_video_path(video_id)
        frames_dir = self.get_frames_dir(video_id)

        if not video_path.exists():
            print(f"Video file `{video_path}` not found")
            return []

        frames_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video `{video_path}` with OpenCV, trying ffmpeg fallback")
            cap.release()
            return self._extract_frames_ffmpeg(video_id, video_path, frames_dir, fps)

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Default fallback

        frame_interval = int(video_fps / fps)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        extracted_count = 0

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps

                # Resize if needed
                height, width = frame.shape[:2]
                if max(height, width) > self.frame_max_dimension:
                    if width > height:
                        new_width = self.frame_max_dimension
                        new_height = int(height * (self.frame_max_dimension / width))
                    else:
                        new_height = self.frame_max_dimension
                        new_width = int(width * (self.frame_max_dimension / height))
                    frame = cv2.resize(frame, (new_width, new_height))

                # Save frame
                frame_filename = f"{video_id}_frame_{extracted_count:05d}.{self.frame_format}"
                frame_path = frames_dir / frame_filename

                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.frame_quality],
                )

                extracted_count += 1
                frames.append({
                    "path": str(frame_path),
                    "time": timestamp,
                    "frame_number": frame_count,
                })
            frame_count += 1

        cap.release()

        if extracted_count == 0:
            print(f"OpenCV extracted 0 frames from `{video_path}` (possible AV1 codec issue), trying ffmpeg fallback")
            return self._extract_frames_ffmpeg(video_id, video_path, frames_dir, fps)

        print(f"Extracted {extracted_count} frames from {video_id}")
        return frames

    def _extract_frames_ffmpeg(self, video_id: str, video_path: Path, frames_dir: Path, fps: float) -> List[dict]:
        """Extract frames using ffmpeg subprocess (handles AV1 and other codecs OpenCV struggles with)."""
        scale_filter = (
            f"scale='if(gt(iw,ih),min({self.frame_max_dimension},iw),-2):"
            f"if(gt(iw,ih),-2,min({self.frame_max_dimension},ih))'"
        )
        output_pattern = str(frames_dir / f"{video_id}_frame_%05d.{self.frame_format}")
        cmd = [
            "ffmpeg",
            "-hwaccel", "none",  # force software decoding to avoid AV1 hw-accel failures
            "-i", str(video_path),
            "-vf", f"fps={fps},{scale_filter}",
            "-q:v", str(max(1, int((100 - self.frame_quality) / 10) + 1)),  # map 0-100 quality to ffmpeg 1-10 scale
            "-start_number", "0",
            output_pattern,
            "-y",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"ffmpeg frame extraction failed for `{video_path}`: {result.stderr}")
                return []
        except FileNotFoundError:
            print("ffmpeg not found; cannot extract frames")
            return []
        except subprocess.TimeoutExpired:
            print(f"ffmpeg timed out extracting frames from `{video_path}`")
            return []

        # Collect saved frames in order and compute timestamps
        saved = sorted(frames_dir.glob(f"{video_id}_frame_*.{self.frame_format}"), key=lambda p: p.name)
        frames = [
            {"path": str(p), "time": round(i / fps, 3), "frame_number": i}
            for i, p in enumerate(saved)
        ]
        print(f"Extracted {len(frames)} frames from {video_id} (via ffmpeg)")
        return frames

    def get_transcript(self, video_id: str, source: Literal["openai", "youtube"] = "openai") -> Optional[dict]:
        """
        Load transcript from file if it exists.

        Args:
            video_id: Video ID
            source: Transcript source ("openai" or "youtube")

        Returns:
            Transcript dict with 'transcript', 'content', and 'transcript_source' fields,
            or None if no transcript of the given source exists.
        """
        transcript_path = None

        if source == "openai":
            transcript_path = self.get_openai_transcript_path(video_id)
        elif source == "youtube":
            transcript_path = self.get_youtube_transcript_path(video_id)

        if transcript_path is not None and transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return None

    def get_video_metadata(self, video_id: str, keys: List[str]) -> Optional[dict]:
        """
        Get video metadata from file if it exists.

        Args:
            video_id: Video ID
            keys: List of keys to get

        Returns:
            Video metadata dict with the given keys, or None if no metadata exists.
        """
        metadata_path = self.get_metadata_path(video_id)
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return {k: metadata.get(k) for k in keys}
        else:
            return None

    def get_video_metadata_summary(self, video_id: str) -> Optional[dict]:
        """
        Get video metadata summary (title, description, creator, tags, duration, release_year, views, likes, dislikes, comments) from file if it exists.

        Args:
            video_id: Video ID

        Returns:
            Video metadata summary dict, or None if no metadata exists.
        """

        summary_keys = ["title", "description", "creator", "tags", "duration", "release_year", "view_count", "like_count", "dislike_count", "comment_count"]

        summary = self.get_video_metadata(video_id, summary_keys)
        if summary is not None:
            return summary
        else:
            return None
