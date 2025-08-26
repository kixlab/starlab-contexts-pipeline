import json

from src import META_TITLE, SIMILARITY_THRESHOLD_STEP, ELEMENTS

from helpers import APPROACHES, BASELINES
from helpers import random_uid

from helpers.clip import clip_similar_per_text, ClipModel
from helpers.bert import clustering_custom, hierarchical_clustering

from helpers.prompts_exp_0 import define_initial_subgoals, extract_subgoals, aggregate_subgoals

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
        self.__establish_subgoal_taxonomy()
        # self.__extract_procedural_elements()

    def __establish_subgoal_taxonomy(self):
        def __subgoal_to_text(subgoal):
            return f"{subgoal['description']}"
        
        # if "subgoal" in self.taxonomies:
        #     return
        if "initial_subgoals" not in self.taxonomies:
            initial_subgoals = define_initial_subgoals(self.task)
            self.taxonomies["initial_subgoals"] = [__subgoal_to_text(subgoal) for subgoal in initial_subgoals]
        initial_subgoals_str = "\n".join(self.taxonomies["initial_subgoals"])
        for video in self.videos:
            if len(video.subgoals) == 0:
                video.subgoals = extract_subgoals(video.get_all_contents(), initial_subgoals_str, self.task)
        
        ### cluster subgoals
        all_subgoals = []
        for video in self.videos:
            all_subgoals.extend(video.subgoals)
        all_subgoals_str = [__subgoal_to_text(subgoal) for subgoal in all_subgoals]
        labels = hierarchical_clustering(all_subgoals_str, embedding_method="bert", linkage="complete", distance_threshold=0.2)

        subgoal_sets = {}
        for i, label in enumerate(labels):
            if label not in subgoal_sets:
                subgoal_sets[label] = []
            subgoal_sets[label].append(all_subgoals[i])
        self.taxonomies["subgoal"] = []
        for subgoal_set in subgoal_sets.values():
            if len(subgoal_set) == 1:
                self.taxonomies["subgoal"].append(subgoal_set[0])
            else:
                subgoal_set_str = [__subgoal_to_text(subgoal) for subgoal in subgoal_set]
                agg_subgoal = aggregate_subgoals(subgoal_set_str, self.task)
                self.taxonomies["subgoal"].append(agg_subgoal)
            
            