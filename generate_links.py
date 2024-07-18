import os
import base64
import random
import json

from PIL import Image
from openai import OpenAI

from generate_data import process_video

PATH = "step_data/"
TASK_ID  = "custom"

LIBRARY = {
    "11967": [
        "https://www.youtube.com/watch?v=yJQShkjNn08",
        "https://www.youtube.com/watch?v=yweUoYP1v_o",
        "https://www.youtube.com/watch?v=Ehntsffsx08",
        "https://www.youtube.com/watch?v=tdk9_Xs_CC0",
        "https://www.youtube.com/watch?v=dkhy4vn9HcY",
        "https://www.youtube.com/watch?v=QECo58lV-bE",
        "https://www.youtube.com/watch?v=SMh2sjuEwxM",
        "https://www.youtube.com/watch?v=DaEzhwLFPi8",
        "https://www.youtube.com/watch?v=J_5scvrv0LU",
        "https://www.youtube.com/watch?v=umbBEHlpTfo",
        "https://www.youtube.com/watch?v=pq_INi_4IBI",
        "https://www.youtube.com/watch?v=pYOQutHfCDo",
    ],
    "custom": [
        # "https://www.youtube.com/shorts/B-XGIGS4Ipw", # short
        # "https://www.youtube.com/shorts/fWp5z_YM07Q", # short
        "https://www.youtube.com/watch?v=aEFvNsBDCWs", # has verbal
        "https://www.youtube.com/watch?v=gN-orgrgvU8", # has verbal
        "https://www.youtube.com/watch?v=cZ2KJPGVwNU", # has verbal
    ]
}

def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

def json_remove_lists(json_obj):
    ### turn all lists into dictionaries with keys as indices `index_0`
    if isinstance(json_obj, list):
        json_obj = {f"index_{i}": json_remove_lists(item) for i, item in enumerate(json_obj)}
    elif isinstance(json_obj, dict):
        for key, value in json_obj.items():
            json_obj[key] = json_remove_lists(value)
    return json_obj

def json_to_markdown(json_obj, indent=0):
    json_obj = json_remove_lists(json_obj)
    
    markdown = ""
    indent_str = "  " * indent
    
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            markdown += f"{indent_str}- **{key}**:\n"
            markdown += json_to_markdown(value, indent + 1)
    else:
        markdown += f"{indent_str}- {json_obj}\n"
    
    return markdown

API_KEY = os.getenv('OPENAI_API_KEY')   
print(API_KEY)
client = OpenAI(
    api_key=API_KEY,
)

SEED = 13774
TEMPERATURE = 0.6
MODEL_NAME = 'gpt-4o'

def get_response(messages, response_format="json_object", retries=1):
    
    generated_text = ""
    finish_reason = ""
    usages = []
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            seed=SEED,
            temperature=TEMPERATURE,
            response_format={
                "type": response_format,
            },

        )
        generated_text += response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usages.append(response.usage)

        if finish_reason != "length":
            break
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

    # print(f"Finish Reason: {finish_reason}")
    # print(f"Usages: {usages}")
    # print(f"Generated Text: {generated_text}")

    if response_format == "json_object":
        try:
            obj = json.loads(generated_text)
            keys = list(obj.keys())
            if len(keys) == 1:
                return obj[keys[0]]
            else:
                return obj
        except json.JSONDecodeError:
            if retries > 0:
                return get_response(messages, response_format, retries - 1)

    return generated_text

def save_data(task_id, videos=None, steps=None):
    if videos is None:
        videos = []
    if steps is None:
        steps = []
    ### save all the video objects
    save_dict = []

    for video in videos:
        save_dict.append(video.to_dict())

    with open(f"{PATH}{task_id}/video_data.json", "w") as file:
        json.dump(save_dict, file, indent=2)

    ### save all the step objects
    with open(f"{PATH}{task_id}/step_data.json", "w") as file:
        json.dump(steps, file, indent=2)

def str_to_float(str_time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))

def float_to_str(float_time):
    return str(int(float_time // 3600)) + ":" + str(int((float_time % 3600) // 60)) + ":" + str(int(float_time % 60))

def generate_custom_steps(subtitles):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes the subtitles of a video into important steps in the procedural task (steps should be based on meaningful intermediate stages of the process). You must use the subtitles to generate the steps. Return a JSON list with the following structure: [{'start': float, 'finish': float, 'title': string, 'text': string}]"},
        {"role": "user", "content": json_to_markdown(subtitles)}
    ]
    return get_response(messages)

def define_common_steps(videos):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that who will define common steps across how-to videos about the same task. The steps should be high-level and based on intermediate stages of the artifact. Make sure that the steps cover all the given narrations in videos. Return the steps and definitions in JSON formart: [{'title': string, 'definition': string}]"},
        {"role": "user", "content": json_to_markdown(videos)}
    ]
    return get_response(messages)

def generate_common_steps(subtitles, common_step_definitions):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who will assist with segmenting the video narration based on given steps. You will use the given narration and step definitions to segment the video. Return a JSON list with the following structure: [{'start': float, 'finish': float, 'title': string, 'text': string, 'explanation': string}]" + f"\nHere are the step definitions: {json_to_markdown(common_step_definitions)}"},
        {"role": "user", "content": json_to_markdown(subtitles)}
    ]
    return get_response(messages)

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


    def __init__(self, video_link):
        self.video_link = video_link
        self.video_id = video_link.split("/")[-1]
        self.subtitles = []
        self.frames = []
        self.custom_steps = []
        self.common_steps = []

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
        self.custom_steps = generate_custom_steps(self.subtitles)
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
        for step in self.custom_steps:
            if timestamp <= step["finish"]:
                return step
        return self.common_steps[-1]

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "video_link": self.video_link,
            "frames": self.frames,
            "subtitles": self.subtitles,
            "custom_steps": self.custom_steps,
            "common_steps": self.common_steps
        }

    ### TODO: Save frames to disk & retrieve based on filename
    def from_dict(self, 
        video_link=None, video_id=None, subtitles=None,
        frames=None, custom_steps=None, common_steps=None
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
        
def pre_process_videos(video_links):
    videos = []
    for video_link in video_links:
        video = Video(video_link)
        video.process()
        videos.append(video)
    return videos


class UnderstandingGap:
    videos = []
    steps = []

    def __init__(self, videos, steps=[]):
        self.videos = videos
        self.steps = steps

    def process_videos(self):
        if len(self.steps) > 0:
            return
        ### Define common steps across all videos
        narrations = []
        for video in self.videos:
            narration = "\n".join([subtitle["text"] for subtitle in video.subtitles])
            narrations.append(narration)
        self.steps = define_common_steps(narrations)
        
        ### Split into common_steps within each video
        for video in self.videos:
            video.common_steps = generate_common_steps(video.custom_steps, self.steps)

    @staticmethod
    def generate_understanding_gaps(previous, next, previous_links, prompt):
        if len(next) == 0:
            return []
        messages = [
            {"role": "system", "content": "You are a helpful expert who knows what kind of understanding gaps learners have about specific tasks. Make sure to be as brief & to the point as possible, avoid verbose sentences. You respond in JSON format."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Follow this JSON format: [{'gap_title': string, 'gap_description': string, 'keywords': list, 'explanation': string, 'video_id': string, 'start': float, 'finish', float}]"},
            {"role": "user", "content": f"Here are the narrations that the learner have already listened & understood:\n{json.dumps(previous)}"},
            {"role": "user", "content": f"Here are the narrations that the learner have not seen yet:\n{json.dumps(next)}"},
            {"role": "user", "content": f"Here are the previous understanding gaps that were identified:\n{json.dumps(previous_links)}"}
        ]

        gaps = get_response(messages)
        ### check if gaps satisfy the format and if not, return an empty list
        if not isinstance(gaps, list):
            return []
        for gap in gaps:
            if not all(key in gap for key in ["gap_title", "gap_description", "keywords", "explanation", "video_id", "start", "finish"]):
                return []
        return gaps

    def transform_watch_history(self, watch_history):
        if len(watch_history) == 0:
            return [], None

        ## combine watch_history of each video under the same video_id
        last_watch = watch_history[-1]
        watch_history_per_video = {}
        for watch in watch_history[:-1]:
            video_id = watch["video_id"]
            if video_id not in watch_history_per_video:
                watch_history_per_video[video_id] = []
            watch_history_per_video[video_id].append(watch)

        previous_with_titles = []
        last_step = None
        for video in self.videos:
            if video.video_id == last_watch["video_id"]:
                last_step_raw = video.get_common_step(last_watch["finish"])
                last_step = {
                    "title": last_step_raw["title"],
                    "video_id": video.video_id,
                    "start": last_step_raw["start"],
                    "finish": last_step_raw["finish"],
                    "text": last_step_raw["text"]
                }

            if video.video_id not in watch_history_per_video:
                continue
            
            steps = video.get_overlapping_steps([(watch["start"], watch["finish"]) for watch in watch_history_per_video[video.video_id]])
            
            if len(steps) > 0:
                for step in steps:
                    previous_with_titles.append({
                        "video_id": video.video_id,
                        "start": step["start"],
                        "finish": step["finish"],
                        "title": step["title"],
                        "text": step["text"]
                    })

        return previous_with_titles, last_step

    def get_global_links(self, previous_with_titles, previous_links=[]):
        previous = []
        covered_video_ids = set()
        for watch in previous_with_titles:
            previous.append({
                "video_id": watch["video_id"],
                "start": watch["start"],
                "finish": watch["finish"],
                "text": watch["text"]
            })
            covered_video_ids.add(watch["video_id"])        

        next = []
        for video in self.videos:
            if video.video_id in covered_video_ids:
                continue
            for step in video.common_steps:
                next.append({
                    "video_id": video.video_id,
                    "start": step["start"],
                    "finish": step["finish"],
                    "text": step["text"],
                })

        prompt = f"""Given (1) the user's watch history (2) the unseen videos, and (3) previous understanding gaps, identify new or old understanding gaps the user can have about the procedure (e.g., order, presence/absence of specific steps, etc) and how they can resolve it by watching the unseen videos. Provide (1) a short title, (2) a short description, (3) few representative keywords, (4) a short explanation how the gap is resolved in the video, and (5) start&finish of the video. Also, make sure to include the understanding gaps that have been resolved yet"""
        
        gaps = self.generate_understanding_gaps(previous, next, previous_links, prompt)
        return gaps

    def get_local_links(self, previous_with_titles, last_step, previous_links=[]):
        previous = []
        covered_video_ids = set()
        for watch in previous_with_titles:
            if watch["title"] != last_step["title"]:
                continue
            previous.append({
                "video_id": watch["video_id"],
                "start": watch["start"],
                "finish": watch["finish"],
                "text": watch["text"]
            })
            covered_video_ids.add(watch["video_id"])

        next = []
        for video in self.videos:
            if video.video_id in covered_video_ids:
                continue
            ## find the step that matches the current step
            for step in video.common_steps:
                if step["title"] == last_step["title"]:
                    next.append({
                        "video_id": video.video_id,
                        "start": step["start"],
                        "finish": step["finish"],
                        "text": step["text"]
                    })
                    break

        prompt = f"""Given (1) the user's watch history (2) the unseen videos, and (3) previous understanding gaps, identify new or old understanding gaps the user can have about the step ({last_step["title"]}) and how they can resolve it by watching the unseen videos. Provide (1) a short title, (2) a short description, (3) few representative keywords, (4) a short explanation how the gap is resolved in the video, and (5) start&finish of the video."""

        gaps = self.generate_understanding_gaps(previous, next, previous_links, prompt)
        return gaps


    def to_dict(self):
        return {
            "videos": [video.to_dict() for video in self.videos],
            "steps": self.steps
        }

def fake_watch_history(videos):
    watch_history = []
    rand_video_idxs = [
        random.randint(0, len(videos) - 1) for i in range(3)
    ]

    for idx in rand_video_idxs:
        video = videos[idx]
        steps = video.common_steps
        rand_watched_idxs = [
            random.randint(0, len(steps) - 1) for i in range(2)
        ]
        for step_idx in rand_watched_idxs:
            step = steps[step_idx]
            watch_history.append({
                "video_id": video.video_id,
                "start": step["start"],
                "finish": step["finish"],
                "title": step["title"],
                "text": step["text"]
            })
    current_step = watch_history.pop()
    return watch_history, current_step

def generate_links(ug, watch_history, link_types = [], previous_links = {}):
    # with open("first_request.json", "r") as f:
    #     data = json.load(f)
    #     old_links = data["links"]
    #     ### do random sampling from old_links & return
    #     links = {}
    #     for link_type in link_types:
    #         links[link_type] = random.sample(old_links[link_type], min(2, len(old_links[link_type])))
    #     return links
    
    if len(watch_history) <= 1:
        ## read `first_request.json` and return the links
        with open("first_request.json", "r") as f:
            data = json.load(f)
            old_links = data["links"]
            old_watch_history = data["watch_history"]
            if (len(watch_history) == 1):
                watch = watch_history[0]
                old_watch = old_watch_history[0]
                same = True
                for key in old_watch.keys():
                    if key not in watch or watch[key] != old_watch[key]:
                        same = False
                        break
                if same is True:
                    return old_links

    previous_with_titles, last_step = ug.transform_watch_history(watch_history)
    links = {}
    if "global" in link_types:
        previous_global = previous_links.get("global", [])
        links["global"] = ug.get_global_links(previous_with_titles, previous_global)
        for link in links["global"]:
            link["label"] = "-Procedure-"
    
    if "local" in link_types and last_step is not None:
        previous_local = previous_links.get("local", [])
        previous_other = [link for link in previous_local if link["label"] != last_step["title"]]
        previous_local = [link for link in previous_local if link["label"] == last_step["title"]]
        links["local"] = ug.get_local_links(previous_with_titles, last_step, previous_local)
        for link in links["local"]:
            link["label"] = f"{last_step['title']}"
        links["local"] = previous_other + links["local"]

    ### save last request
    open("last_request.json", "w").write(json.dumps({
        "watch_history": watch_history,
        "link_types": link_types,
        "previous_with_titles": previous_with_titles,
        "last_step": last_step,
        "previous_links": previous_links,
        "links": links,
    }, indent=2))

    ### Do this on the frontend
    # if last_step is not None:
    #     ## generate local links
    #     ## links that current_video resolves
    #     links_self = []
    #     for link_type in links.keys():
    #         links_self += [link for link in links[link_type] if link["video_id"] == last_step["video_id"]]
            
    #         links[link_type] = [link for link in links[link_type] if link["video_id"] != last_step["video_id"]]
        
    #         links[link_type] = sorted(links[link_type], key=lambda x: x["start"])
        
    #     links_self = sorted(links_self, key=lambda x: x["start"])
    #     links["self"] = links_self
    return links

def setup_ug(task_id):
    if task_id not in LIBRARY:
        return None

    # get the video data
    videos = []
    steps = []
    video_data_path = f"{PATH}{task_id}/video_data.json"
    step_data_path = f"{PATH}{task_id}/step_data.json"

    if os.path.exists(video_data_path):
        with open(video_data_path, "r") as file:
            video_data = json.load(file)
            videos = []
            for data in video_data:
                if data["video_link"] not in LIBRARY[task_id]:
                    continue
                video = Video(data["video_link"])
                video.from_dict(**data)
                videos.append(video)
    if len(videos) == 0 or len(videos[0].subtitles) == 0:
        videos = pre_process_videos(LIBRARY[task_id])
        save_data(task_id, videos=videos)

    if os.path.exists(step_data_path):
        with open(step_data_path, "r") as file:
            steps = json.load(file)
        ug = UnderstandingGap(videos, steps)
    else:
        ug = UnderstandingGap(videos)
        ug.process_videos()
        save_data(task_id, videos=ug.videos, steps=ug.steps)
    return ug