from backend import process_video

from helpers import str_to_float

class Video:
    video_link = ""
    video_id = None
    ### list of frames in base64 {"idx": 0, "image": "", caption: ""}
    frames = []
    ### {"start": 0, "finish": 0, "text": ""}
    subtitles = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    custom_steps = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    common_steps = []

    meta_summary = None
    subgoal_summaries = []


    def __init__(self, video_link):
        self.video_link = video_link
        self.video_id = video_link.split("/")[-1]
        self.subtitles = []
        self.frames = []
        self.custom_steps = []
        self.common_steps = []
        self.meta_summary = None
        self.subgoal_summaries = []

    def process(self):
        self.process_video()
        self.process_subtitles()

    def process_video(self):
        video_title, video_frame_paths, subtitles = process_video(self.video_link)
        self.video_id = video_title
        self.frames = []
        
        for idx, frame_path in enumerate(video_frame_paths):
            self.frames.append({
                "idx": idx,
                "path": frame_path,
            })
        
        self.subtitles = []    
        for subtitle in subtitles:
            self.subtitles.append({
                "start": str_to_float(subtitle["start"]),
                "finish": str_to_float(subtitle["finish"]),
                "text": subtitle["text"]
            })

    def process_subtitles(self):
        ## TODO: reevaluate if custom steps are needed
        # self.custom_steps = generate_custom_steps(self.subtitles)
        self.custom_steps = []
        for subtitle in self.subtitles:
            self.custom_steps.append({
                "start": subtitle["start"],
                "finish": subtitle["finish"],
                # "title": "",
                "text": subtitle["text"]
            })
        ### sort the steps by start time
        self.custom_steps = sorted(self.custom_steps, key=lambda x: x["start"])

    def get_overlapping_steps(self, segments):
        # segments = [(start, finish)] 
        if len(self.common_steps) == 0:
            return []
        steps = []
        for step in self.common_steps:
            ## check if it overlaps with any of the segments
            for start, finish in segments:
                if max(start, step["start"]) < min(finish, step["finish"]):
                    steps.append(step)
                    break
        return steps
    
    def get_common_step(self, timestamp):
        if len(self.common_steps) == 0:
            return None
        for step in self.common_steps:
            if step["start"] <= timestamp <= step["finish"]:
                return step
        for step in self.common_steps:
            if timestamp <= step["finish"]:
                return step
        return self.common_steps[-1]
    
    def get_full_narration(self):
        return "\n".join([subtitle["text"] for subtitle in self.subtitles])
    
    def find_subgoal_summary(self, title):
        for summary in self.subgoal_summaries:
            if summary["title"] == title:
                return summary
        return None

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "video_link": self.video_link,
            "frames": self.frames,
            "subtitles": self.subtitles,
            "custom_steps": self.custom_steps,
            "common_steps": self.common_steps,
            "meta_summary": self.meta_summary,
            "subgoal_summaries": self.subgoal_summaries,
        }

    ### TODO: Save frames to disk & retrieve based on filename
    def from_dict(self, 
        video_link=None, video_id=None, subtitles=None,
        frames=None, custom_steps=None, common_steps=None,
        meta_summary=None, subgoal_summaries=None
    ):
        if video_link is not None:
            self.video_link = video_link
        if video_id is not None:
            self.video_id = video_id
        if subtitles is not None:
            self.subtitles = subtitles
        if frames is not None:
            self.frames = frames
        if custom_steps is not None:
            self.custom_steps = custom_steps
        if common_steps is not None:
            self.common_steps = common_steps
        if meta_summary is not None:
            self.meta_summary = meta_summary
        if subgoal_summaries is not None:
            self.subgoal_summaries = subgoal_summaries