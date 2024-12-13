import json

from src import META_TITLE, SIMILARITY_THRESHOLD_STEP

from helpers import APPROACHES, BASELINES
from helpers import random_uid

from helpers.clip import clip_similar_per_text, ClipModel
from helpers.bert import clustering_custom

from helpers.prompts_step import get_initial_steps, get_steps, aggregate_steps

class VideoPool:
    task = ""
    videos = []
    taxonomies = {}

    def __init__(self, task, videos, taxonomies={}):
        self.task = task
        self.videos = videos
        self.taxonomies = taxonomies

    def get_video(self, video_id):
        for video in self.videos:
            if video.video_id == video_id:
                return video
        return None

    def process_videos(self):
        self.__establish_step_taxonomy()


    ### Stage 1: Establish step taxonomy
    def __establish_step_taxonomy(self):
        if "step" in self.taxonomies:
            return
        taxonomy = get_initial_steps(self.task)
        taxonomy_contents = []
        for step in taxonomy:
            taxonomy_contents.append({
                "text": f"{step['title']}: {step['description']}",
                "frame_paths": [],
            })
        for video in self.videos:
            ### update the taxonomy
            video.steps = get_steps(video.get_all_contents(), taxonomy_contents)
            
        all_steps = []
        for v_index, video in enumerate(self.videos):
            for s_index, step in enumerate(video.steps):
                all_steps.append({
                    "text": step["title"],
                    "v_index": v_index,
                    "s_index": s_index,
                })
        ### cluster the similar steps
        labels = clustering_custom(
            [step["text"] for step in all_steps],
            SIMILARITY_THRESHOLD_STEP
        )            

        ### update the taxonomy
        steps_per_label = {}
        for i, label in enumerate(labels):
            if label not in steps_per_label:
                steps_per_label[label] = []
            steps_per_label[label].append(all_steps[i])
        
        new_taxonomy = []
        for label in steps_per_label:
            steps = steps_per_label[label]
            agg_step = steps[0]
            if len(steps) > 1:
                step_contents = []
                for step in steps:
                    video_step = self.videos[step["v_index"]].steps[step["s_index"]]
                    step_contents.append({
                        "text": f"{video_step['title']}: {video_step['description']}",
                        "frame_paths": [],
                    })
                agg_step = aggregate_steps(step_contents)
            for step in steps:
                v_index = step["v_index"]
                s_index = step["s_index"]
                self.videos[v_index].steps[s_index] = {
                    **self.videos[v_index].steps[s_index],
                    "title": agg_step["title"],
                    "description": agg_step["description"],
                }
            new_taxonomy.append(agg_step)
        
        ### combine neighboring similar steps
        # for video in self.videos:
        #     new_steps = []
        #     for step in video.steps:
        #         if len(new_steps) == 0 or new_steps[-1]["title"] != step["title"]:
        #             new_steps.append(step)
        #         else:
        #             new_steps[-1][""]
        #     video.steps = new_steps

        ### save taxonomy
        self.taxonomies["step"] = new_taxonomy
        return new_taxonomy

        