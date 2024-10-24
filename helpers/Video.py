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
    ### list of strings
    steps = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    sentences = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    subgoals = []

    meta_summary = None
    subgoal_summaries = []

    sentence_embeddings = []


    def __init__(self, video_link):
        self.video_link = video_link
        self.video_id = video_link.split("/")[-1]
        self.subtitles = []
        self.steps = []
        self.frames = {}
        self.sentences = []
        self.subgoals = []
        self.meta_summary = None
        self.subgoal_summaries = []
        self.metadata = {}
        self.sentence_embeddings = []

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
        self.sentences = []
        for subtitle in self.subtitles:
            self.sentences.append({
                "start": subtitle["start"],
                "finish": subtitle["finish"],
                "text": subtitle["text"],
                "frame_paths": [],
            })
        
        self.sentences = sorted(self.sentences, key=lambda x: x["start"])

        for index, sentence in enumerate(self.sentences):
            frame_sec = round((sentence["start"] + sentence["finish"]) / 2)
            if frame_sec in self.frames:
                sentence["frame_paths"].append(self.frames[frame_sec]["path"])
            sentence["id"] = f"{self.video_id}-{index}"
    
    def get_subgoals(self, title):
        subgoals = []
        for subgoal in self.subgoals:
            if subgoal["title"] == title:
                subgoals.append(subgoal)
        return subgoals
    
    def get_full_narration(self):
        return "\n".join([subtitle["text"] for subtitle in self.subtitles])
    
    def get_all_contents(self):
        contents = []
        for sentence in self.sentences:
            contents.append({
                "id": sentence["id"],
                "start": sentence["start"],
                "finish": sentence["finish"],
                "text": sentence["text"],
                "frame_paths": [path for path in sentence["frame_paths"]],
            })
        return contents

    def get_meta_summary_contents(self, as_context=False) -> list:
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
        subgoals = self.get_subgoals(title)
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
                "id": subgoal["content_ids"][0],
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
        if len(self.sentence_embeddings) == 0:
            self.calculate_sentence_embeddings()
        content_ids = []
        quotes_embeddings = bert_embedding(quotes)

        indexes, scores = find_most_similar(self.sentence_embeddings, quotes_embeddings)
        for idx in indexes:
            content_ids.append(self.sentences[idx]["id"])

        return content_ids
                        
    def calculate_sentence_embeddings(self):
        """
        Calculate the embeddings of the custom subgoals
        """
        texts = [sentence["text"] for sentence in self.sentences]
        self.sentence_embeddings = bert_embedding(texts)

    def get_most_similar_content_ids(self, texts):
        """
        Returns the content ids of the most similar texts
        """
        if len(texts) == 0:
            return []
        if len(self.sentence_embeddings) == 0:
            self.calculate_sentence_embeddings()
        text_embeddings = bert_embedding(texts)

        indexes, scores = find_most_similar(self.sentence_embeddings, text_embeddings)

        content_ids = []
        for idx in indexes:
            content_ids.append(self.sentences[idx]["id"])
        return content_ids

    def to_dict(self, short_metadata=False, fixed_subgoals=False):
        result = {
            "video_id": self.video_id,
            "video_link": self.video_link,
            "frames": self.frames,
            "subtitles": self.subtitles,
            "sentences": self.sentences,
            "steps": self.steps,
            "subgoals": self.subgoals,
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
            for index, subgoal in enumerate(result["subgoals"]):
                subgoal["original_title"] = subgoal["title"]
                subgoal["title"] = subgoal["original_title"] + "-" + str(index)
            
            for index, subgoal_summary in enumerate(result["subgoal_summaries"]):
                for subgoal in result["subgoals"]:
                    if subgoal_summary["title"] == subgoal["original_title"]:
                        subgoal_summary["original_title"] = subgoal["original_title"]
                        subgoal_summary["title"] = subgoal["title"]
        return result

    def from_dict(self, 
        video_link=None, video_id=None, subtitles=None,
        frames=None, sentences=None, steps=None, subgoals=None,
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
        if sentences is not None:
            self.sentences = sentences
        if steps is not None:
            self.steps = steps
        if subgoals is not None:
            self.subgoals = subgoals
        if meta_summary is not None:
            self.meta_summary = meta_summary
        if subgoal_summaries is not None:
            self.subgoal_summaries = subgoal_summaries
        if metadata is not None:
            self.metadata = metadata