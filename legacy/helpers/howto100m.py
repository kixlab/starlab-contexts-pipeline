import os
import webvtt
import whisper
import cv2
import csv
import json
import re

from pathlib import Path

from yt_dlp import YoutubeDL

DATABASE = Path("static/database")
HOWTO = Path("howto100m")

class Video:
    video_id = ""
    video_link = ""
    processed = False
    video_path = ""
    subtitle_path = ""
    audio_path = ""
    metadata = {}

    def __init__(self, video_id, video_link):
        self.video_id = video_id
        self.video_link = video_link
        self.processed = False

    def process(self):
        options = {
            'format': 'bv[height<=?480][ext=mp4]+ba[ext=mp3]/best',
            #'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=mp3]/best',
            'outtmpl': os.path.join(DATABASE, '%(id)s.%(ext)s'),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': {'en'},  # Download English subtitles
            'subtitlesformat': '/vtt/g',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'keepvideo': True,
            'skip_download': False,
        }

        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(self.video_link, download=False)
            self.metadata = ydl.sanitize_info(info)
            video_title = self.metadata.get('id')
            video_path = os.path.join(DATABASE, f'{video_title}.mp4')
            if not os.path.exists(video_path):
                ydl.download([self.video_link])
                print(f"Video '{video_title}' downloaded successfully.")
            else:
                print(f"Video '{video_title}' already exists in the directory.")

            self.video_path = video_path
            self.subtitle_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
            self.audio_path = os.path.join(DATABASE, f'{video_title}.mp3')
            self.extract_transcript_list()
            self.processed = True
            return self.metadata

    def extract_frames(self):
        video_cap = cv2.VideoCapture(self.video_path)
        
        frames = []
        
        while (True):
            res, frame = video_cap.read()
            if (res == False):
                break

            frames.append(frame)
        
        video_cap.release()

        return frames


    def extract_transcript_from_audio(self):
        output_path = self.audio_path.replace(".mp3", ".alt.json")
        raw_transcript = {}
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                raw_transcript = json.load(f)
        else:
            model = whisper.load_model("small.en")
            raw_transcript = model.transcribe(self.audio_path)
            with open(output_path, 'w') as f:
                json.dump(raw_transcript, f, indent=2)

        transcript = []
        for segment in raw_transcript["segments"]:
            transcript.append({
                "start": segment["start"],
                "finish": segment["end"],
                "text": segment["text"],
            })
        return transcript

    def extract_transcript_list(self):
        if not os.path.exists(self.subtitle_path):
            print(f"Subtitles file '{self.subtitle_path}' does not exist.")
            if not os.path.exists(self.audio_path):
                print(f"Audio file '{self.audio_path}' does not exist.")
                return []
            transcript = self.extract_transcript_from_audio(self.audio_path)

            return transcript

        subtitles = webvtt.read(self.subtitle_path)

        transcript = []
        for caption in subtitles:
            lines = caption.text.strip("\n ").split("\n")
            if len(transcript) == 0:
                transcript.append({
                    "start": caption.start,
                    "finish": caption.end,
                    "text": "\n".join(lines),
                })
                continue
            last_caption = transcript[len(transcript) - 1]

            new_text = ""
            for line in lines:
                if line.startswith(last_caption["text"], 0):
                    new_line = line[len(last_caption["text"]):-1].strip()
                    if len(new_line) > 0:
                        new_text += new_line + "\n"
                elif len(line) > 0:
                    new_text += line + "\n"
            new_text = new_text.strip("\n ")
            if len(new_text) == 0:
                transcript[len(transcript) - 1]["finish"] = caption.end
            else:
                transcript.append({
                    "start": caption.start,
                    "finish": caption.end,
                    "text": new_text,
                })
        return transcript
    
    def extract_transcript_str(self):
        transcript = self.extract_transcript_list()
        transcript_str = "\n".join([f"{seg['text']} " for seg in transcript])
        # remove new lines
        transcript_str = transcript_str.replace("\n", " ")
        # remove any extra spaces
        transcript_str = re.sub(' +', ' ', transcript_str)

        return transcript_str

    def __str__(self):
        result = f"Video: {self.video_id}\n"
        ### print title, description, tags, etc.
        if not self.processed:
            return result
        result += f"\tTitle: {self.metadata['title']}\n"
        result += f"\tDescription: {self.metadata['tags']}; {self.metadata['categories']}\n"
        result += f"\tTranscript:\n"
        result += self.extract_transcript_str()
        "----------------------------------------------------------------\n"
        result += "\n"
        return result
    
def process_howto100m():
    video_ids_path = HOWTO / "HowTo100M_v1.csv"
    task_ids_path = HOWTO / "task_ids.csv"

    videos_info = []
    with open(video_ids_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            videos_info.append({
                "video_id": row["video_id"],
                "cat1": row["category_1"],
                "cat2": row["category_2"],
                "rank": row["rank"],
                "task_id": row["task_id"],
            })
    
    task_id_titles = {}
    with open(task_ids_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            task_id_titles[row["id"]] = row["task_name"]
    
    for video_info in videos_info:
        video_info["task_name"] = task_id_titles[video_info["task_id"]]

    return videos_info

def process_videos(
    n=100,
    task_id="11967",
    cat1="Food and Entertaining",
    cat2="Recipes"
):
    ### read csv file of HowTo100M dataset
    videos = []
    
    videos_info = process_howto100m()

    for video_info in videos_info:
        if (video_info["cat1"] == cat1 
            and video_info["cat2"] == cat2
            and video_info["task_id"] == task_id
        ):
            video = Video(video_info["video_id"], f"https://www.youtube.com/watch?v={video_info['video_id']}")
            try:
                video.process()
                videos.append(video)
            except Exception as e:
                print(f"Failed to process video '{video_info['video_id']}': {e}")
                continue
        if len(videos) >= n:
            break
    print(f"Number of videos: {len(videos)}")
    return videos

def json_dump_video_links(videos):
    json_dump = []
    for idx, video in enumerate(videos):
        json_dump.append(video.video_link)
    
    print(json.dumps(json_dump, indent=2))
    return