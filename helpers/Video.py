from helpers.video_scripts import process_video

from helpers import str_to_float

class Video:
    video_link = ""
    video_id = None
    ### list of frames in base64 sec: {"path": "", caption: ""}
    frames = {}
    ### {"start": 0, "finish": 0, "text": ""}
    subtitles = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    custom_subgoals = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    common_subgoals = []

    meta_summary = None
    subgoal_summaries = []


    def __init__(self, video_link):
        self.video_link = video_link
        self.video_id = video_link.split("/")[-1]
        self.subtitles = []
        self.frames = {}
        self.custom_subgoals = []
        self.common_subgoals = []
        self.meta_summary = None
        self.subgoal_summaries = []

    def process(self):
        self.process_video()
        self.process_subtitles()

    def process_video(self):
        video_title, video_frame_paths, subtitles = process_video(self.video_link)
        self.video_id = video_title
        self.frames = {}
        
        for sec, frame_path in enumerate(video_frame_paths):
            self.frames[sec] = {
                "path": frame_path,
                "caption": "",
            }
        
        self.subtitles = []    
        for subtitle in subtitles:
            self.subtitles.append({
                "start": subtitle["start"],
                "finish": subtitle["finish"],
                "text": subtitle["text"]
            })

    def process_subtitles(self):
        self.subtitles = sorted(self.subtitles, key=lambda x: x["start"])

        self.custom_subgoals = []
        for subtitle in self.subtitles:
            # if len(self.custom_subgoals) > 0:
            #     last_subgoal = self.custom_subgoals[-1]
            #     if subtitle['finish'] - last_subgoal['start'] <= 10:
            #         last_subgoal['finish'] = subtitle['finish']
            #         last_subgoal['text'] += " " + subtitle['text']
            #         continue
            self.custom_subgoals.append({
                "start": subtitle["start"],
                "finish": subtitle["finish"],
                "text": subtitle["text"],
                "frame_paths": [],
            })
        
        self.custom_subgoals = sorted(self.custom_subgoals, key=lambda x: x["start"])

        for subgoal in self.custom_subgoals:
            frame_sec = round((subgoal["start"] + subgoal["finish"]) / 2)
            if frame_sec in self.frames:
                subgoal["frame_paths"].append(self.frames[frame_sec]["path"])

    def get_overlapping_subgoals(self, segments):
        # segments = [(start, finish)] 
        if len(self.common_subgoals) == 0:
            return []
        subgoals = []
        for subgoal in self.common_subgoals:
            ## check if it overlaps with any of the segments
            for start, finish in segments:
                if max(start, subgoal["start"]) < min(finish, subgoal["finish"]):
                    subgoals.append(subgoal)
                    break
        return subgoals
    
    def get_common_subgoal(self, timestamp):
        if len(self.common_subgoals) == 0:
            return None
        for subgoal in self.common_subgoals:
            if subgoal["start"] <= timestamp <= subgoal["finish"]:
                return subgoal
        for subgoal in self.common_subgoals:
            if timestamp <= subgoal["finish"]:
                return subgoal
        return self.common_subgoals[-1]
    
    def get_full_narration(self):
        return "\n".join([subtitle["text"] for subtitle in self.subtitles])
    
    def get_all_contents(self):
        contents = []
        for index, subgoal in enumerate(self.custom_subgoals):
            contents.append({
                "id": f"content-{index}",
                "start": subgoal["start"],
                "finish": subgoal["finish"],
                "text": subgoal["text"],
                "frame_paths": [path for path in subgoal["frame_paths"]],
            })
        contents = sorted(contents, key=lambda x: x["start"])
        return contents
    
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
            "custom_subgoals": self.custom_subgoals,
            "common_subgoals": self.common_subgoals,
            "meta_summary": self.meta_summary,
            "subgoal_summaries": self.subgoal_summaries,
        }

    ### TODO: Save frames to disk & retrieve based on filename
    def from_dict(self, 
        video_link=None, video_id=None, subtitles=None,
        frames=None, custom_subgoals=None, common_subgoals=None,
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
        if custom_subgoals is not None:
            self.custom_subgoals = custom_subgoals
        if common_subgoals is not None:
            self.common_subgoals = common_subgoals
        if meta_summary is not None:
            self.meta_summary = meta_summary
        if subgoal_summaries is not None:
            self.subgoal_summaries = subgoal_summaries