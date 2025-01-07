import json

from src import META_TITLE, SIMILARITY_THRESHOLD_STEP, ELEMENTS

from helpers import APPROACHES, BASELINES
from helpers import random_uid

from helpers.clip import clip_similar_per_text, ClipModel
from helpers.bert import clustering_custom, hierarchical_clustering

from helpers.prompts_step import get_initial_steps, get_steps, aggregate_steps

from helpers.prompts_tax import extract_procedural_elements, aggregate_procedural_elements

from helpers.prompts_experimental import get_instructions

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
        #if "step" not in self.taxonomies:
        # self.__establish_step_taxonomy()
        # self.__extract_procedural_elements()
        # self.__assign_procedural_elements()
        self.__extract_instructions()

    def __extract_instructions(self):
        for video in self.videos:
            video.steps = get_instructions(video.get_all_contents(), self.task)

    def __cluster_procedural_elements(self, element_type, elements, similarity_thresh, method="llm"):
        elements_text = [element["description"] for element in elements]
        # labels = clustering_custom(elements_text, similarity_thresh, embedding_method="bert")
        labels = hierarchical_clustering(elements_text, embedding_method="bert", linkage="complete", distance_threshold=1.0-similarity_thresh)
        elements_per_label = {}
        for i, label in enumerate(labels):
            if label not in elements_per_label:
                elements_per_label[label] = []
            elements_per_label[label].append(elements[i])

        agg_elements = []
        if method == "llm":
            for label in elements_per_label:
                elements = elements_per_label[label]
                if len(elements) == 0:
                    continue
                if len(elements) == 1:
                    agg_elements.append({
                        **elements[0],
                    })
                    continue
                element_contents = []
                for element in elements:
                    element_contents.append({
                        "text": element["description"],
                        "frame_paths": [],
                    })
                agg_element = aggregate_procedural_elements(element_contents, element_type, self.task)
                agg_elements.append(agg_element)
        else:
            for label in elements_per_label:
                elements = elements_per_label[label]
                if len(elements) == 0:
                    continue
                agg_elements.append({
                    **elements[0],
                })
        return agg_elements

    def __extract_procedural_elements(self):
        all_elements = {}
        for element_type in ELEMENTS:
            all_elements[element_type] = []
        for video in self.videos:
            input, step, output = extract_procedural_elements(video.get_all_contents(), self.task)
            all_elements[ELEMENTS[0]] += input
            all_elements[ELEMENTS[1]] += step
            all_elements[ELEMENTS[2]] += output
        
        ### cluster inputs, steps, outcomes
        #SIMILARITY_THRESH_LIST = [0.6, 0.75, 0.85]
        SIMILARITY_THRESH_LIST = [0.8]

        for similarity_thresh in SIMILARITY_THRESH_LIST:
            suffix = str(similarity_thresh).split(".")[1]
            for element_type in ELEMENTS:
                elements = all_elements[element_type]
                agg_elements = self.__cluster_procedural_elements(
                    element_type, elements,
                    similarity_thresh, method="llm")
                self.taxonomies[element_type + "_" + suffix] = agg_elements
        return self.taxonomies
    
    def __assign_procedural_elements(self):
        ### process the videos based on the taxonomy!
        ### Almost VQA-like approach
        low_level_taxonomies = {
            "input": self.taxonomies["input_80"],
            "step": self.taxonomies["step_80"],
            "output": self.taxonomies["output_80"],
        }
        for element_type in ELEMENTS:
            low_level_tax = low_level_taxonomies[element_type]
            tax_contents = [
                {
                    "text": element["title"],
                    "frame_paths": [],
                }
                for element in low_level_tax
            ]
            # for video in self.videos:
                

    # ### Stage 1: Establish step taxonomy
    # def __establish_step_taxonomy(self):
    #     taxonomy = get_initial_steps(self.task)
    #     taxonomy_contents = []
    #     for step in taxonomy:
    #         taxonomy_contents.append({
    #             "text": f"{step['title']}: {step['description']}",
    #             "frame_paths": [],
    #         })
    #     for video in self.videos:
    #         ### update the taxonomy
    #         video.steps = get_steps(video.get_all_contents(), taxonomy_contents, self.task)
            
    #     all_steps = []
    #     for v_index, video in enumerate(self.videos):
    #         for s_index, step in enumerate(video.steps):
    #             all_steps.append({
    #                 "text": step["title"],
    #                 "v_index": v_index,
    #                 "s_index": s_index,
    #             })
    #     ### cluster the similar steps
    #     labels = clustering_custom(
    #         [step["text"] for step in all_steps],
    #         SIMILARITY_THRESHOLD_STEP
    #     )            

    #     ### update the taxonomy
    #     steps_per_label = {}
    #     for i, label in enumerate(labels):
    #         if label not in steps_per_label:
    #             steps_per_label[label] = []
    #         steps_per_label[label].append(all_steps[i])
        
    #     new_taxonomy = []
    #     for label in steps_per_label:
    #         steps = steps_per_label[label]
    #         agg_step = {}
    #         if len(steps) > 1:
    #             step_contents = []
    #             for step in steps:
    #                 video_step = self.videos[step["v_index"]].steps[step["s_index"]]
    #                 step_contents.append({
    #                     "text": f"{video_step['title']}: {video_step['description']}",
    #                     "frame_paths": [],
    #                 })
    #             agg_step = aggregate_steps(step_contents, self.task)
    #         else:
    #             video_step = self.videos[steps[0]["v_index"]].steps[steps[0]["s_index"]]
    #             agg_step = {
    #                 "title": video_step["title"],
    #                 "description": video_step["description"],
    #             }
    #         for step in steps:
    #             v_index = step["v_index"]
    #             s_index = step["s_index"]
    #             self.videos[v_index].steps[s_index] = {
    #                 **self.videos[v_index].steps[s_index],
    #                 "title": agg_step["title"],
    #                 "description": agg_step["description"],
    #             }
    #         new_taxonomy.append(agg_step)
        
    #     ### combine neighboring similar steps
    #     # for video in self.videos:
    #     #     new_steps = []
    #     #     for step in video.steps:
    #     #         if len(new_steps) == 0 or new_steps[-1]["title"] != step["title"]:
    #     #             new_steps.append(step)
    #     #         else:
    #     #             new_steps[-1][""]
    #     #     video.steps = new_steps

    #     ### save taxonomy
    #     self.taxonomies["step"] = new_taxonomy
    #     return new_taxonomy

        