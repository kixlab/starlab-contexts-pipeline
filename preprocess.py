import os
import json

from helpers import PATH, LIBRARY, TASK_DESCRIPTIONS, META_TITLE, VIDEO_SETS
from helpers.Video import Video
from helpers.llm_prompting import define_common_steps, generate_common_steps, get_meta_summary, get_subgoal_summary, get_meta_alignments, get_subgoal_alignments, get_alignment_classification, get_hooks
from helpers.mdprint import print_hooks


def save_data(task_id, videos=None, steps=None, alignments=None, hooks=None):
    if videos is None:
        videos = []
    if steps is None:
        steps = []
    if alignments is None:
        alignments = []
    if hooks is None:
        hooks = {}
    ### save all the video objects
    save_dict = []

    for video in videos:
        save_dict.append(video.to_dict())
    
    if os.path.exists(f"{PATH}{task_id}") is False:
        os.mkdir(f"{PATH}{task_id}")

    with open(f"{PATH}{task_id}/video_data.json", "w") as file:
        json.dump(save_dict, file, indent=2)

    ### save all the step objects
    with open(f"{PATH}{task_id}/step_data.json", "w") as file:
        json.dump(steps, file, indent=2)

    ### save all the information alignments
    with open(f"{PATH}{task_id}/alignments.json", "w") as file:
        json.dump(alignments, file, indent=2)

    ### save all the hooks
    with open(f"{PATH}{task_id}/hooks.json", "w") as file:
        json.dump(hooks, file, indent=2)
        
def pre_process_videos(video_links):
    videos = []
    for video_link in video_links:
        video = Video(video_link)
        try:
            video.process()
            videos.append(video)
        except Exception as e:
            print(f"Error processing video: {video_link}")
            print(e)
            continue
    return videos

### DYNAMIC SUMMARIES

class DynamicSummary:
    task = ""
    videos = []
    steps = []

    alignments = []

    hooks = {}

    def __init__(self, task, videos, steps=[]):
        self.task = task
        self.videos = videos
        self.steps = steps

    def process_videos(self):
        if len(self.steps) == 0:
            ### Define common steps across all videos
            narrations = []
            for video in self.videos:
                narration = "\n".join([subtitle["text"] for subtitle in video.subtitles])
                narrations.append(narration)
            common_steps = define_common_steps(narrations)
            for step in common_steps:
                self.steps.append({
                    "title": step["title"],
                    "definition": step["definition"],
                    "dependencies": step["dependencies"] if "dependencies" in step else [],
                    "explanation": step["explanation"] if "explanation" in step else "",
                })

        ### Split into common_steps within each video
        for video in self.videos:
            if len(video.common_steps) == 0:
                cur_common_steps = generate_common_steps(video.custom_steps, self.steps)
                for step in cur_common_steps:
                    video.common_steps.append({
                        "title": step["title"],
                        "start": step["start"],
                        "finish": step["finish"],
                        "text": step["text"],
                        "explanation": step["explanation"] if "explanation" in step else "",
                    })
            
            ## Summarize (for each video) (1) problem, (2) method, (3) outcome
            if video.meta_summary is None:
                video.meta_summary = get_meta_summary(META_TITLE, video.get_full_narration())

            ## Summarize (for each subgoal): (1) context, (2) tools, (3) instructions, (4) explanations, (5) supplementary info, (6) outcome
            if len(video.subgoal_summaries) == 0:
                for main_step_def in self.steps:
                    relevant_narrations = []
                    parent_schemas = []
                    for parent_summary in video.subgoal_summaries:
                        if parent_summary["title"] in main_step_def["dependencies"]:
                            parent_schemas.append(parent_summary)
                    
                    for step in video.common_steps:
                        if step["title"] == main_step_def["title"]:
                            relevant_narrations.append(step)
                    
                    video.subgoal_summaries.append(get_subgoal_summary(main_step_def["title"], relevant_narrations, parent_schemas, video.meta_summary))
        
        if len(self.videos) >= 2 and len(self.alignments) == 0:
            self.generate_alignments()

    def generate_alignments(self):
        self.alignments = []

        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                ### between meta
                meta_alignments = get_meta_alignments(video1.meta_summary, video2.meta_summary)
                self.alignments.append({
                    **meta_alignments,
                    "title": META_TITLE,
                    "new_video": video1.video_id,
                    "prev_video": video2.video_id,
                })
                ### between subgoals
                for step in self.steps:
                    summary1 = video1.find_subgoal_summary(step["title"])
                    if summary1 is None:
                        continue    
                    summary2 = video2.find_subgoal_summary(step["title"])
                    if summary2 is None:
                        continue
                    subgoal_alignments = get_subgoal_alignments(step["title"], video1.meta_summary, video2.meta_summary, summary1, summary2)
                    self.alignments.append({
                        **subgoal_alignments,
                        "title": step["title"],
                        "new_video": video1.video_id,
                        "prev_video": video2.video_id,
                    })
    
    def classify_alignments(self):
        for video_pair in self.alignments:
            title = video_pair["title"]
            alignments = video_pair["alignments"]
            new_video = video_pair["new_video"]
            prev_video = video_pair["prev_video"]
            for video in self.videos:
                if new_video == video.video_id:
                    new_video = video
                if prev_video == video.video_id:
                    prev_video = video
            
            new_meta = None
            prev_meta = None
            new_subgoal = None
            prev_subgoal = None

            if title == META_TITLE:
                new_meta = new_video.meta_summary
                prev_meta = prev_video.meta_summary
            else:
                new_meta = new_video.meta_summary
                prev_meta = prev_video.meta_summary
                new_subgoal = new_video.find_subgoal_summary(title)
                prev_subgoal = prev_video.find_subgoal_summary(title)
            for index, alignment in enumerate(alignments):
                if "classification" in alignment:
                    continue
                classification = get_alignment_classification(alignment, title, new_meta, prev_meta, new_subgoal, prev_subgoal)
                alignments[index] = {
                    **alignment,
                    **classification
                }

    def genenerate_all_hooks(self):
        ### TODO: may need a slightly different approach --> generate hooks for the entire video considering all the alignments regardless of subgoals and then reclassify the hooks into each subgoal!!!
        ### generate hooks to "seen" videos
        if "seen" not in self.hooks:
            alignments_per_video = {}
            for video_pair in self.alignments:
                new_video = video_pair["new_video"]
                prev_video = video_pair["prev_video"]
                title = video_pair["title"]
                if new_video not in alignments_per_video:
                    alignments_per_video[new_video] = {}
                if title not in alignments_per_video[new_video]:
                    alignments_per_video[new_video][title] = []
                for alignment in video_pair["alignments"]:
                    alignments_per_video[new_video][title].append({
                        **alignment,
                        "prev_video": prev_video,
                    })
            

            self.hooks["seen"] = {} # video_id -> title -> class -> hooks
            for new_video, alignments_per_title in alignments_per_video.items():
                self.hooks["seen"][new_video] = {}
                ### clarification: we combine meta into each "subgoal-level" hook generation!!!!
                meta_alignments = []
                if META_TITLE in alignments_per_title:
                    meta_alignments = alignments_per_title[META_TITLE]
                    for index, alignment in enumerate(meta_alignments):
                        alignment["alignment_id"] = f"m-{index}"
                for title, alignments in alignments_per_title.items():
                    if title == META_TITLE:
                        continue
                    for index, alignment in enumerate(alignments):
                        alignment["alignment_id"] = f"a-{index}"
                    hooks = self.generate_hooks(VIDEO_SETS["seen"], title, meta_alignments + alignments)
                    self.hooks["seen"][new_video][title] = hooks
        
        ### generate hooks to "unseen" videos
        if "unseen" not in self.hooks:
            ### filter out the set of hooks to be combined "per video"
            hooks_per_video = {}
            for new_video, hooks_per_title in self.hooks["seen"].items():
                for title, hooks_per_class in hooks_per_title.items():
                    for classification, hooks in hooks_per_class.items():
                        for hook in hooks:
                            for alignment in hook["alignments"]:
                                prev_video = alignment["prev_video"]
                                if prev_video not in hooks_per_video:
                                    hooks_per_video[prev_video] = {}
                                if title not in hooks_per_video[prev_video]:
                                    hooks_per_video[prev_video][title] = {}
                                if classification not in hooks_per_video[prev_video][title]:
                                    hooks_per_video[prev_video][title][classification] = []
                                hooks_per_video[prev_video][title][classification].append(hook)

            self.hooks["unseen"] = {} # video_id -> title -> class -> hooks
            for prev_video, hooks_per_title in hooks_per_video.items():
                self.hooks["unseen"][prev_video] = {}
                for title, hooks_per_class in hooks_per_title.items():
                    self.hooks["unseen"][prev_video][title] = {}
                    for classification, hooks in hooks_per_class.items():
                        self.hooks["unseen"][prev_video][title][classification] = self.__combine_hooks(title, hooks)

    def __combine_hooks(self, title, hooks):
        ### TODO: combine hooks with similar newref / description!!!
        return hooks
    
    def generate_hooks(self, video_set, title, original_alignments):
        alignments_per_class = {}
        for alignment in original_alignments:
            classification = alignment["classification"]
            if classification not in alignments_per_class:
                alignments_per_class[classification] = []
            
            alignments_per_class[classification].append({
                "alignment_id": alignment["alignment_id"],
                "description": alignment["description"],
                "provenance": alignment["provenance"],
                "newref": alignment["newref"],
                "prevref": alignment["prevref"],
            })
        hooks_per_class = {}
        for classification, alignments in alignments_per_class.items():
            hooks = self.__generate_hooks(video_set, classification, title, alignments)
            hooks_per_class[classification] = []
            for hook in hooks:
                alignment_ids = [alignment["alignment_id"] for alignment in hook["alignments"]]
                hooks_per_class[classification].append({
                    **hook,
                    "alignments": [alignment for alignment in original_alignments if alignment["alignment_id"] in alignment_ids]
                })
        return hooks_per_class
    
    def __generate_hooks(self, video_set, classification, title, alignments):
        hooks = get_hooks(video_set, classification, title, alignments)

        covered_alignment_ids = []
        for hook in hooks:
            cur_alignment_ids = [alignment["alignment_id"] for alignment in hook["alignments"]]
            covered_alignment_ids += cur_alignment_ids

        uncovered_alignments = []
        has_non_meta = False
        for alignment in alignments:
            if alignment["alignment_id"].startswith("m-") is False:
                has_non_meta = True
            if alignment["alignment_id"] not in covered_alignment_ids:
                uncovered_alignments.append(alignment)
        
        if len(uncovered_alignments) > 0 and has_non_meta:
            hooks.extend(self.__generate_hooks(video_set, classification, title, uncovered_alignments))
        return hooks

def setup_ds(task_id):
    if task_id not in LIBRARY:
        return None

    # get the video data
    videos = []
    steps = []
    alignments = []
    video_data_path = f"{PATH}{task_id}/video_data.json"
    step_data_path = f"{PATH}{task_id}/step_data.json"
    alignments_path = f"{PATH}{task_id}/alignments.json"
    hooks_path = f"{PATH}{task_id}/hooks.json"

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

    task_desc = TASK_DESCRIPTIONS[task_id]

    if os.path.exists(step_data_path):
        with open(step_data_path, "r") as file:
            steps = json.load(file)
        ds = DynamicSummary(task_desc, videos, steps)
    else:
        ds = DynamicSummary(task_desc, videos)

    if os.path.exists(alignments_path):
        with open(alignments_path, "r") as file:
            alignments = json.load(file)
            ds.alignments = alignments
    else:
        ds.alignments = []

    ds.process_videos()

    ds.classify_alignments()

    ds.genenerate_all_hooks()

    save_data(task_id, videos=ds.videos, steps=ds.steps, alignments=ds.alignments, hooks=ds.hooks)
    return ds

def main():

    ds = setup_ds("custom2")

if __name__ == "__main__":
    main()