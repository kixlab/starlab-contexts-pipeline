from helpers.video_scripts import process_video

from helpers import str_to_float, segment_into_sentences

from helpers.bert import bert_embedding, find_most_similar

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

    custom_subgoals_embeddings = []


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
        self.subtitles = sorted(self.subtitles, key=lambda x: x["start"])

    def process_subtitles(self):
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
        return contents

    def get_meta_summary_contents(self) -> list:
        ### TODO: If going bad, may need to add quotes
        if self.meta_summary is None:
            return None
        quotes = {}
        for k, v in self.meta_summary.items():
            if k.endswith("_quotes"):
                quotes[k[:-7]] = v
        
        text = ""
        for k, v in self.meta_summary.items():
            if k not in quotes:
                continue
            key = "Overall " + k.capitalize().replace("_", " ")
            value = v if isinstance(v, str) else ", ".join(v)
            text += f"- **{key}**: {value}\n"
            if len(quotes[k]) > 0:
                text += f"\t- **{key} Quotes**:"
                text += "; ".join([f"`{quote}`" for quote in quotes[k]])
                text += "\n"
        return [{
            "id": f"meta-summary",
            "text": text,
            "frame_paths": [path for path in self.meta_summary["frame_paths"]],
        }]

    def get_subgoal_summary_contents(self, title, as_parent=False) -> list:
        for index, summary in enumerate(self.subgoal_summaries):
            if summary["title"] != title:
                continue
            text = ""
            if as_parent:
                ### indicate that this is a parent
                text += f"- **Parent Subgoal**: {summary['title']}\n"
            else:
                ### add the contents
                text += f"- **Subgoal Contents**:\n"
            quotes = {}
            for k, v in summary.items():
                if k.endswith("_quotes"):
                    quotes[k[:-7]] = v
            for k, v in summary.items():
                if k not in quotes:
                    continue
                if as_parent and k != "outcome":
                    continue
                key = k.capitalize().replace("_", " ")
                value = v if isinstance(v, str) else ", ".join(v)
                text += f"\t- **{key}**: {value}\n"
                if len(quotes[k]) > 0 and not as_parent:
                    text += f"\t- **{key} Quotes**:"
                    text += "; ".join([f"`{quote}`" for quote in quotes[k]])
                    text += "\n"
            content = {
                "id": f"subgoal-summary-{index}",
                "title": summary["title"],
                "text": text,
                "frame_paths": [path for path in summary["frame_paths"]],
            }
            if as_parent:
                return [content]
            return [content for content in summary["context"]] + [content]
        return []

    def get_subgoal_contents(self, title, as_parent=False) -> list:
        for index, subgoal in enumerate(self.common_subgoals):
            if subgoal["title"] != title:
                continue
            text = ""
            if as_parent:
                ### indicate that this is a parent
                text += f"#### Parent Subgoal {subgoal['title']}\n"
            else:
                ### add the contents
                text += f"#### Subgoal Contents**\n"
            
            text += subgoal["text"] + "\n"
            return [{
                "id": f"content-{index}",
                "text": text,
                "frame_paths": [path for path in subgoal["frame_paths"]],
            }]
        return []
    
    def quotes_to_content_ids(self, quotes):
        """
        Returns the content ids of the quotes
        """
        if len(quotes) == 0:
            return []
        contents = self.get_all_contents()
        if len(self.custom_subgoals_embeddings) == 0:
            self.calculate_custom_subgoals_embeddings()
        content_ids = []
        quotes_embeddings = bert_embedding(quotes)

        indexes = find_most_similar(self.custom_subgoals_embeddings, quotes_embeddings)
        for idx in indexes:
            content_ids.append(contents[idx]["id"])

        return content_ids

                        
    def calculate_custom_subgoals_embeddings(self):
        """
        Calculate the embeddings of the custom subgoals
        """
        texts = [subgoal["text"] for subgoal in self.custom_subgoals]
        self.custom_subgoals_embeddings = bert_embedding(texts)

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