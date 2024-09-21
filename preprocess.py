import os
import json

from helpers import PATH, LIBRARY, TASK_DESCRIPTIONS, META_TITLE, VIDEO_SETS
from helpers.Video import Video
from helpers.llm_prompting_v1 import define_common_subgoals_v1, generate_common_subgoals_v1, get_meta_summary_v1, get_subgoal_summary_v1, get_meta_alignments_v1, get_subgoal_alignments_v1, get_alignment_classification_v1, get_hooks_v1

from helpers.llm_prompting_v2 import define_common_subgoals_v2, align_common_subgoals_v2, get_meta_summary_v2, get_subgoal_summary_v2, generate_common_subgoals_v3, get_subgoal_alignments_v2, get_meta_alignments_v2

from helpers.clip import clip_similar_per_text
from helpers.mdprint import print_hooks


def save_data(task_id, ds):
    videos = ds.videos
    subgoals = ds.subgoals
    alignments = ds.alignments
    alignments_baseline_1 = ds.alignments_baseline_1
    alignments_baseline_2 = ds.alignments_baseline_2
    hooks = ds.hooks

    ### save all the video objects
    save_dict = []

    for video in videos:
        save_dict.append(video.to_dict())
    
    if os.path.exists(f"{PATH}{task_id}") is False:
        os.mkdir(f"{PATH}{task_id}")

    with open(f"{PATH}{task_id}/video_data.json", "w") as file:
        json.dump(save_dict, file, indent=2)

    ### save all the subgoal objects
    with open(f"{PATH}{task_id}/subgoal_data.json", "w") as file:
        json.dump(subgoals, file, indent=2)

    ### save all the information alignments
    with open(f"{PATH}{task_id}/alignments.json", "w") as file:
        json.dump(alignments, file, indent=2)

    ### save all the information alignments
    with open(f"{PATH}{task_id}/alignments_baseline_1.json", "w") as file:
        json.dump(alignments_baseline_1, file, indent=2)

    ### save all the information alignments
    with open(f"{PATH}{task_id}/alignments_baseline_2.json", "w") as file:
        json.dump(alignments_baseline_2, file, indent=2)

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
    subgoals = []

    alignments = []
    alignments_baseline_1 = []
    alignments_baseline_2 = []

    hooks = {}

    def __init__(self, task, videos, subgoals=[]):
        self.task = task
        self.videos = videos
        self.subgoals = subgoals

    def process_videos(self):
        ### feed each narration separately to identify common subgoals
        # self.__process_videos_v1()
        ### first split the videos into subgoals and then align the subgoals to each other and redefine them
        self.__process_videos_v2()

    def __process_videos_v1(self):
        if len(self.subgoals) == 0:
            ### Define common subgoals across all videos
            narrations = []
            for video in self.videos:
                narration = "\n".join([subtitle["text"] for subtitle in video.subtitles])
                narrations.append(narration)
            common_subgoals = define_common_subgoals_v1(narrations)
            for subgoal in common_subgoals:
                self.subgoals.append({
                    "title": subgoal["title"],
                    "definition": subgoal["definition"],
                    "dependencies": subgoal["dependencies"] if "dependencies" in subgoal else [],
                    "explanation": subgoal["explanation"] if "explanation" in subgoal else "",
                })

        ### Split into common_subgoals within each video
        for video in self.videos:
            if len(video.common_subgoals) == 0:
                cur_common_subgoals = generate_common_subgoals_v1(video.subtitles, self.subgoals)
                for subgoal in cur_common_subgoals:
                    video.common_subgoals.append({
                        "title": subgoal["title"],
                        "start": subgoal["start"],
                        "finish": subgoal["finish"],
                        "text": subgoal["text"],
                        "explanation": subgoal["explanation"] if "explanation" in subgoal else "",
                    })
            
            ## Summarize (for each video) (1) context, (2) method, (3) outcome
            if video.meta_summary is None:
                video.meta_summary = get_meta_summary_v1(META_TITLE, video.get_full_narration())

            ## Summarize (for each subgoal): (1) context, (2) tools, (3) instructions, (4) explanations, (5) supplementary info, (6) outcome
            if len(video.subgoal_summaries) == 0:
                for main_subgoal_def in self.subgoals:
                    relevant_narrations = []
                    parent_schemas = []
                    for parent_summary in video.subgoal_summaries:
                        if parent_summary["title"] in main_subgoal_def["dependencies"]:
                            parent_schemas.append(parent_summary)
                    
                    for subgoal in video.common_subgoals:
                        if subgoal["title"] == main_subgoal_def["title"]:
                            relevant_narrations.append(subgoal)
                    
                    video.subgoal_summaries.append(get_subgoal_summary_v1(main_subgoal_def["title"], relevant_narrations, parent_schemas, video.meta_summary))

    def __process_videos_v2(self):
        if len(self.subgoals) == 0:
            common_subgoal_defs_per_video = {}
            for video in self.videos:
                if len(video.common_subgoals) > 0:
                    continue
                common_subgoal_defs_per_video[video.video_id] = define_common_subgoals_v2(video.get_all_contents(), self.task)

            self.subgoals = align_common_subgoals_v2(common_subgoal_defs_per_video, self.task)

        for video in self.videos:
            if len(video.common_subgoals) == 0:
                common_subgoals = generate_common_subgoals_v3(video.get_all_contents(), self.subgoals, self.task)
                ### `segmentation` has `text`, `quotes`, `explanation`
                for subgoal in common_subgoals:
                    video.common_subgoals.append({
                        "title": subgoal["title"],
                        "explanation": subgoal["explanation"],
                        "start": subgoal["start"],
                        "finish": subgoal["finish"],
                        "text": subgoal["text"],
                        "frame_paths": subgoal["frame_paths"],
                    })
            
            ## Summarize (for each video) (1) context, (2) method, (3) outcome
            if video.meta_summary is None:
                summary = get_meta_summary_v2(video.get_all_contents(), self.task)
                video.meta_summary = {
                    "title": META_TITLE,
                    **summary,
                }
                # ASSUMPTION: The video has a single major outcome
                video.meta_summary["frame_paths"] = clip_similar_per_text([video.meta_summary["outcome"]], video.common_subgoals[-1]["frame_paths"])

                extension = {}
                ### turn _quotes into content ids
                for key in video.meta_summary:
                    if key.endswith("_quotes"):
                        new_key = key.replace("_quotes", "_content_ids")
                        extension[new_key] = video.quotes_to_content_ids(video.meta_summary[key])
                video.meta_summary = {
                    **video.meta_summary,
                    **extension
                }

            # Summarize (for each subgoal)
            if len(video.subgoal_summaries) == 0:
                for subgoal_def in self.subgoals:
                    contents = []
                    for subgoal in video.common_subgoals:
                        if subgoal["title"] == subgoal_def["title"]:
                            contents.append({
                                "text": subgoal["text"],
                                "frame_paths": subgoal["frame_paths"],
                            })
                    
                    if len(contents) == 0:
                        continue

                    context = video.get_meta_summary_contents()
                    for parent_title in subgoal_def["dependencies"]:
                        context += video.get_subgoal_summary_contents(parent_title, True)
                    subgoal_summary = get_subgoal_summary_v2(contents, context, subgoal_def["title"], self.task)
                    summary = {
                        "title": subgoal_def["title"],
                        **subgoal_summary,
                    }
                    summary["frame_paths"] = clip_similar_per_text([summary["outcome"]], summary["frame_paths"])
                    
                    extension = {}
                    for key in summary:
                        if key.endswith("_quotes"):
                            new_key = key.replace("_quotes", "_content_ids")
                            extension[new_key] = video.quotes_to_content_ids(summary[key])
                    summary = {
                        **summary,
                        **extension,
                        "context": context[1:],
                    }
                    video.subgoal_summaries.append(summary)

    def generate_alignments(self):
        if len(self.videos) < 2 or len(self.alignments) > 0:
            return
        if len(self.subgoals) == 0:
            return
        self.alignments = []
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                ### between subgoals
                for subgoal_def in self.subgoals:
                    contents1 = video1.get_subgoal_summary_contents(subgoal_def["title"])
                    contents2 = video2.get_subgoal_summary_contents(subgoal_def["title"])
                    if len(contents1) == 0 or len(contents2) == 0:
                        continue
                    subgoal_alignments = get_subgoal_alignments_v2(
                        contents1, contents2, subgoal_def["title"], self.task
                    )
                    for alignment in subgoal_alignments:
                        ### convert `cur_quotes` and `prev_quotes` to content ids
                        alignment["cur_content_ids"] = video1.quotes_to_content_ids(alignment["cur_quotes"])
                        alignment["prev_content_ids"] = video2.quotes_to_content_ids(alignment["prev_quotes"])
                    self.alignments.append({
                        "alignments": subgoal_alignments,
                        "title": subgoal_def["title"],
                        "video": video1.video_id,
                        "prev_video": video2.video_id,
                    })
    
    def generate_alignments_baseline_1(self):
        if len(self.videos) < 2 or len(self.alignments_baseline_1) > 0:
            return
        if len(self.subgoals) == 0:
            return
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                ### between subgoals
                for subgoal_def in self.subgoals:
                    contents1 = video1.get_subgoal_contents(subgoal_def["title"])
                    contents2 = video2.get_subgoal_contents(subgoal_def["title"])
                    if len(contents1) == 0 or len(contents2) == 0:
                        continue
                    context1 = []
                    context2 = []
                    for parent_title in subgoal_def["dependencies"]:
                        context1 += video1.get_subgoal_contents(parent_title, True)
                        context2 += video2.get_subgoal_contents(parent_title, True)
                    subgoal_alignments = get_subgoal_alignments_v2(
                        context1 + contents1, context2 + contents2, subgoal_def["title"], self.task
                    )
                    for alignment in subgoal_alignments:
                        ### convert `cur_quotes` and `prev_quotes` to content ids
                        alignment["cur_content_ids"] = video1.quotes_to_content_ids(alignment["cur_quotes"])
                        alignment["prev_content_ids"] = video2.quotes_to_content_ids(alignment["prev_quotes"])

                    self.alignments_baseline_1.append({
                        "alignments": subgoal_alignments,
                        "title": subgoal_def["title"],
                        "cur_video": video1.video_id,
                        "prev_video": video2.video_id,
                    })
    
    def generate_alignments_baseline_2(self):
        if len(self.videos) < 2 or len(self.alignments_baseline_2) > 0:
            return
        if len(self.subgoals) == 0:
            return
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                contents1 = video1.get_all_contents()
                contents2 = video2.get_all_contents()
                if len(contents1) == 0 or len(contents2) == 0:
                    continue
                subgoal_alignments = get_meta_alignments_v2(
                    contents1, contents2, self.task
                )
                for alignment in subgoal_alignments:
                    ### convert `cur_quotes` and `prev_quotes` to content ids
                    alignment["cur_content_ids"] = video1.quotes_to_content_ids(alignment["cur_quotes"])
                    alignment["prev_content_ids"] = video2.quotes_to_content_ids(alignment["prev_quotes"])

                self.alignments_baseline_2.append({
                    "alignments": subgoal_alignments,
                    "title": META_TITLE,
                    "cur_video": video1.video_id,
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
                classification = get_alignment_classification_v1(alignment, title, new_meta, prev_meta, new_subgoal, prev_subgoal)
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
        hooks = get_hooks_v1(video_set, classification, title, alignments)

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
    subgoals = []
    alignments = []
    video_data_path = f"{PATH}{task_id}/video_data.json"
    subgoal_data_path = f"{PATH}{task_id}/subgoal_data.json"
    alignments_path = f"{PATH}{task_id}/alignments.json"
    alignments_baseline_1_path = f"{PATH}{task_id}/alignments_baseline_1.json"
    alignments_baseline_2_path = f"{PATH}{task_id}/alignments_baseline_2.json"
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

    if os.path.exists(subgoal_data_path):
        with open(subgoal_data_path, "r") as file:
            subgoals = json.load(file)
        ds = DynamicSummary(task_desc, videos, subgoals)
    else:
        ds = DynamicSummary(task_desc, videos)

    if os.path.exists(alignments_path):
        with open(alignments_path, "r") as file:
            alignments = json.load(file)
            ds.alignments = alignments
    else:
        ds.alignments = []

    if os.path.exists(alignments_baseline_1_path):
        with open(alignments_baseline_1_path, "r") as file:
            alignments = json.load(file)
            ds.alignments_baseline_1 = alignments
    else:
        ds.alignments_baseline_1 = []

    if os.path.exists(alignments_baseline_2_path):
        with open(alignments_baseline_2_path, "r") as file:
            alignments = json.load(file)
            ds.alignments_baseline_2 = alignments
    else:
        ds.alignments_baseline_2 = []

    ds.process_videos()
    
    ds.generate_alignments()
    ds.generate_alignments_baseline_1()
    ds.generate_alignments_baseline_2()
    
    # ds.classify_alignments()
    # ds.genenerate_all_hooks()

    save_data(task_id, ds)
    return ds

def main():

    # for task_id in TASK_DESCRIPTIONS.keys():
    #     ds = setup_ds(task_id)
    #     for video in ds.videos:
    #         for subgoal in video.common_subgoals:
    #             print(subgoal["title"], end=";")
    #         print()

    task_id = list(TASK_DESCRIPTIONS.keys())[0]
    ds = setup_ds(task_id)

if __name__ == "__main__":
    main()