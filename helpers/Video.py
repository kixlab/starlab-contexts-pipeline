from helpers.video_scripts import process_video

from helpers import str_to_float, segment_into_sentences

from helpers.bert import bert_embedding, find_most_similar

class Video:
    video_link = ""
    metadata = {}
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
        video_title, video_frame_paths, subtitles, metadata = process_video(self.video_link)
        self.metadata = metadata
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

        for index, subgoal in enumerate(self.custom_subgoals):
            frame_sec = round((subgoal["start"] + subgoal["finish"]) / 2)
            if frame_sec in self.frames:
                subgoal["frame_paths"].append(self.frames[frame_sec]["path"])
            subgoal["id"] = f"{self.video_id}-{index}",
    
    def get_common_subgoals(self, title):
        subgoals = []
        for subgoal in self.common_subgoals:
            if subgoal["title"] == title:
                subgoals.append(subgoal)
        return subgoals
    
    def get_full_narration(self):
        return "\n".join([subtitle["text"] for subtitle in self.subtitles])
    
    def get_all_contents(self):
        contents = []
        for subgoal in self.custom_subgoals:
            contents.append({
                "id": subgoal["id"],
                "start": subgoal["start"],
                "finish": subgoal["finish"],
                "text": subgoal["text"],
                "frame_paths": [path for path in subgoal["frame_paths"]],
            })
        return contents

    def get_meta_summary_contents(self, as_context=False) -> list:
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
            if len(quotes[k]) > 0 and not as_context:
                text += f"\t- **{key} Quotes**:"
                text += "; ".join([f"`{quote}`" for quote in quotes[k]])
                text += "\n"
        return [{
            "id": f"{self.video_id}-meta",
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
                "id": f"{self.video_id}-subgoal-summary-{index}",
                "title": summary["title"],
                "text": text,
                "frame_paths": [path for path in summary["frame_paths"]],
            }
            if as_parent:
                return [content]
            return [content for content in summary["context"]] + [content]
        return []

    def get_subgoal_contents(self, title, as_parent=False) -> list:
        contents = []
        subgoals = self.get_common_subgoals(title)
        for subgoal in subgoals:
            text = ""
            if as_parent:
                ### indicate that this is a parent
                text += f"#### Parent Subgoal {subgoal['title']}\n"
            else:
                ### add the contents
                text += f"#### Subgoal Contents**\n"
            
            text += subgoal["text"] + "\n"
            contents.append({
                "id": subgoal['id'],
                "text": text,
                "frame_paths": [path for path in subgoal["frame_paths"]],
                "content_ids": subgoal["content_ids"]
            })
        return contents
    
    def quotes_to_content_ids(self, quotes):
        """
        Returns the content ids of the quotes
        """
        if len(quotes) == 0:
            return []
        if len(self.custom_subgoals_embeddings) == 0:
            self.calculate_custom_subgoals_embeddings()
        content_ids = []
        quotes_embeddings = bert_embedding(quotes)

        indexes, scores = find_most_similar(self.custom_subgoals_embeddings, quotes_embeddings)
        for idx in indexes:
            content_ids.append(self.custom_subgoals[idx]["id"])

        return content_ids
                        
    def calculate_custom_subgoals_embeddings(self):
        """
        Calculate the embeddings of the custom subgoals
        """
        texts = [subgoal["text"] for subgoal in self.custom_subgoals]
        self.custom_subgoals_embeddings = bert_embedding(texts)

    def get_most_similar_content_ids(self, texts):
        """
        Returns the content ids of the most similar texts
        """
        if len(texts) == 0:
            return []
        if len(self.custom_subgoals_embeddings) == 0:
            self.calculate_custom_subgoals_embeddings()
        text_embeddings = bert_embedding(texts)

        indexes, scores = find_most_similar(self.custom_subgoals_embeddings, text_embeddings)

        content_ids = []
        for idx in indexes:
            content_ids.append(self.custom_subgoals[idx]["id"])
        return content_ids

    def to_dict(self, short_metadata=False, fixed_subgoals=False):
        result = {
            "video_id": self.video_id,
            "video_link": self.video_link,
            "frames": self.frames,
            "subtitles": self.subtitles,
            "custom_subgoals": self.custom_subgoals,
            "common_subgoals": self.common_subgoals,
            "meta_summary": self.meta_summary,
            "subgoal_summaries": self.subgoal_summaries,
            "metadata": self.metadata
        }
        if short_metadata:
            result["metadata"] = {
                "title": self.metadata["title"],
                "duration": self.metadata["duration"],
                "width": self.metadata["width"],
                "height": self.metadata["height"],
                "fps": self.metadata["fps"],
            }
        
        if fixed_subgoals:
            for index, subgoal in enumerate(result["common_subgoals"]):
                subgoal["original_title"] = subgoal["title"]
                subgoal["title"] = subgoal["original_title"] + "-" + str(index)
            
            for index, subgoal_summary in enumerate(result["subgoal_summaries"]):
                for subgoal in result["common_subgoals"]:
                    if subgoal_summary["title"] == subgoal["original_title"]:
                        subgoal_summary["original_title"] = subgoal["original_title"]
                        subgoal_summary["title"] = subgoal["title"]
        return result

    def from_dict(self, 
        video_link=None, video_id=None, subtitles=None,
        frames=None, custom_subgoals=None, common_subgoals=None,
        meta_summary=None, subgoal_summaries=None, metadata=None
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
        if metadata is not None:
            self.metadata = metadata