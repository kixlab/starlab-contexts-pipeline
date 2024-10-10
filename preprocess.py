import os
import json

from helpers import PATH, LIBRARY, TASK_DESCRIPTIONS, META_TITLE
from helpers import random_uid, paraphrase

from helpers.Video import Video

from helpers.llm_prompting_v2 import define_common_subgoals_v2, align_common_subgoals_v2, get_meta_summary_v2, get_subgoal_summary_v2, generate_common_subgoals_v3, get_subgoal_alignments_v2, get_meta_alignments_v2, get_alignments_summary_v2, get_notable_v2, get_hook_v2

from helpers.llm_prompting_v3 import get_subgoal_alignments_v3, get_meta_alignments_v3

from helpers.clip import clip_similar_per_text
from helpers.sklearn import cluster_tagged_texts, cluster_texts
from helpers.bert import bert_embedding, find_most_similar

APPROACHES = [
    "approach_1",
    "approach_2",
]

BASELINES = [
    "baseline_1",
    "baseline_2",
]


def save_data(task_id, ds):
    videos = ds.videos
    subgoals = ds.subgoals
    alignment_sets = ds.alignment_sets
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
    with open(f"{PATH}{task_id}/alignment_sets.json", "w") as file:
        json.dump(alignment_sets, file, indent=2)

    # ### save all the information alignments
    # with open(f"{PATH}{task_id}/alignments.json", "w") as file:
    #     json.dump(alignments, file, indent=2)

    # ### save all the information alignments
    # with open(f"{PATH}{task_id}/alignments_baseline_1.json", "w") as file:
    #     json.dump(alignments_baseline_1, file, indent=2)

    # ### save all the information alignments
    # with open(f"{PATH}{task_id}/alignments_baseline_2.json", "w") as file:
    #     json.dump(alignments_baseline_2, file, indent=2)

    ### save all the hooks
    with open(f"{PATH}{task_id}/hooks.json", "w") as file:
        json.dump(hooks, file, indent=2)


def export(task_id, ds):    
    save_data(task_id, ds)

    videos = [video.to_dict(short_metadata=True, fixed_subgoals=True) for video in ds.videos]

    def fix_subgoals(items):
        for item in items:
            seconds = item["seconds"]
            new_subgoal = None
            for video in videos:
                if video["video_id"] == item["video_id"]:
                    for subgoal in video["common_subgoals"]:
                        if subgoal["start"] <= seconds and subgoal["finish"] >= seconds:
                            new_subgoal = subgoal["original_title"]
                            break
                    if new_subgoal is None:
                        new_subgoal = META_TITLE
                    break
            item["subgoal"] = new_subgoal
        return items

    def find_seconds(video_id, content_ids, is_link):
        video = None
        for v in ds.videos:
            if v.video_id == video_id:
                video = v
                break
        if video is None:
            return None
        
        contents = video.get_all_contents()
        content_counter = {}
        for content_id in content_ids:
            index = int(content_id.split("-")[-1])
            if index >= len(contents):
                continue
            if index not in content_counter:
                content_counter[index] = 0
            content_counter[index] += 1

        max_count = 0
        max_index = -1
        for index, count in content_counter.items():
            if count > max_count:
                max_count = count
                max_index = index
            elif count == max_count:
                if is_link:
                    if index < max_index:
                        max_index = index
                else:
                    if index > max_index:
                        max_index = index
        if max_index == -1:
            return contents[-1]["finish"]
        
        content = contents[max_index]
        if is_link:
            return content["start"]
        return content["finish"]

    def fix_alignments(items, video_id):
        new_items = []
        for item in items:
            title = None
            description = None
            comparison = None
            reasoning = None
            content_ids = []
            other_video_id = None
            other_content_ids = []
            subgoal = None

            if "notable_title" in item:
                content_ids = item["other_content_ids"]
                title = item["notable_title"]
                description = item["notable_description"]
                other_video_id = item["video_id"]
                other_content_ids = item["notable_content_ids"]
                comparison = item["alignment_comparison"] ## TODO: previous -> current
                if "alignment_reasoning" in item:
                    reasoning = item["alignment_reasoning"] ## TODO: previous -> current
            else:
                content_ids = item["content_ids"]
                title = item["alignment_title"]
                description = item["alignment_description"]
                other_video_id = item["other_video_id"]
                other_content_ids = item["other_content_ids"]
                comparison = item["alignment_comparison"]
                if "alignment_reasoning" in item:
                    reasoning = item["alignment_reasoning"]
            
            tag = None
            if 'classification' in item and item['classification'] is not None and item['classification'] != "":
                tag = item['classification']

            id = item["id"]
            subgoal = item["subgoal_title"]
            other_seconds = find_seconds(other_video_id, other_content_ids, True)
            seconds = find_seconds(video_id, content_ids, True)
            
            new_items.append({
                "id": id,
                "video_id": video_id,
                "title": title,
                "description": description,
                "comparison": comparison,
                "reasoning": reasoning,
                "content_ids": content_ids,
                "other_video_id": other_video_id,
                "other_content_ids": other_content_ids,
                "subgoal": subgoal,
                "seconds": seconds,
                "other_seconds": other_seconds,
                "tag": tag,
            })
        new_items = fix_subgoals(new_items)
        return new_items

    def fix_hooks(items, label):
        new_items = []
        for item in items:
            video_id = item["video_id"]
            title = None
            description = None
            content_ids = []
            tag = None
            links = []
            parent_id = None
            subgoal = None

            if label == "hook":
                title = item["hook_title"]
                description = item["hook_description"]
                content_ids = item["hook_content_ids"]
                links = fix_alignments(item["links"], video_id)
            if label == "notable":
                title = item["notable_title"]
                description = item["notable_description"]
                content_ids = item["notable_content_ids"]
                links = fix_alignments(item["alignments"], video_id)

            if 'classification' in item and item['classification'] is not None and item['classification'] != "":
                tag = item['classification']

            id = item["id"]
            subgoal = item["subgoal_title"]
            parent_id = item["parent_id"]
            seconds = find_seconds(video_id, content_ids, False)
            
            new_items.append({
                "id": id,
                "label": label,
                "video_id": video_id,
                "title": title,
                "description": description,
                "content_ids": content_ids,
                "tag": tag,
                "seconds": seconds,
                "links": links,
                "subgoal": subgoal,
                "parent_id": parent_id,
            })
        new_items = fix_subgoals(new_items)
        return new_items

    def fix_raw_alignments(items):
        new_items = []
        for item in items:
            label = "raw"
            video_id = item["video_id"]
            title = "Raw Alignment"
            description = "Raw Alignment"
            tag = None

            if 'classification' in item and item['classification'] is not None and item['classification'] != "":
                tag = item['classification']
            
            content_ids = []
            for a in item["alignments"]:
                content_ids += a["content_ids"]
            links = fix_alignments(item["alignments"], video_id)
            
            seconds = find_seconds(video_id, content_ids, False)
            subgoal = META_TITLE
            parent_id = None

            id = f"{label}-{video_id}-{random_uid()}"
            new_items.append({
                "id": id,
                "label": label,
                "video_id": video_id,
                "title": title,
                "description": description,
                "content_ids": content_ids,
                "tag": tag,
                "seconds": seconds,
                "links": links,
                "subgoal": subgoal,
                "parent_id": parent_id,
            })
        new_items = fix_subgoals(new_items)
        return new_items

    output = {
        "task": ds.task,
        "videos": videos,
        "subgoal_definitions": ds.subgoals,
        "hooks": {}
    }

    for approach in APPROACHES:
        if approach in ds.alignment_sets:
            output["hooks"][approach] = (fix_hooks(ds.hooks[f"hooks_{approach}"], "hook") +
            fix_hooks(ds.hooks[f"notables_{approach}"], "notable") +
            fix_raw_alignments(ds.alignment_sets[approach]))

    filename = f"{PATH}{task_id}/output.json"
    with open(filename, "w") as file:
        json.dump(output, file, indent=2)
        
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

    alignment_sets = {}

    hooks = {}

    def __init__(self, task, videos, subgoals=[]):
        self.task = task
        self.videos = videos
        self.subgoals = subgoals

    def get_video(self, video_id):
        for video in self.videos:
            if video.video_id == video_id:
                return video
        return None

    def process_videos(self):
        self.__process_videos_v2()

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
                for index, subgoal in enumerate(common_subgoals):
                    video.common_subgoals.append({
                        "id": f"{video.video_id}-subgoal-{index}",
                        "title": subgoal["title"],
                        "explanation": subgoal["explanation"],
                        "start": subgoal["start"],
                        "finish": subgoal["finish"],
                        "text": subgoal["text"],
                        "frame_paths": subgoal["frame_paths"],
                        "content_ids": subgoal["content_ids"],
                    })
            
            ## Summarize (for each video) (1) context, (2) method, (3) outcome
            if video.meta_summary is None:
                summary = get_meta_summary_v2(video.get_all_contents(), self.task)
                video.meta_summary = {
                    "title": META_TITLE,
                    **summary,
                }
                last_subgoals = []
                subgoal_idx = len(self.subgoals) - 1
                while len(last_subgoals) == 0:
                    last_subgoals = video.get_common_subgoals(self.subgoals[subgoal_idx]["title"])
                    subgoal_idx -= 1
                if len(last_subgoals) == 0:
                    last_subgoals = video.common_subgoals
                # ASSUMPTION: The video has a single major outcome
                video.meta_summary["frame_paths"] = clip_similar_per_text([video.meta_summary["outcome"]], last_subgoals[-1]["frame_paths"])

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
                    contents = video.get_subgoal_contents(subgoal_def["title"])
                    if len(contents) == 0:
                        continue

                    context = video.get_meta_summary_contents(True)
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
                        "context": context,
                    }
                    video.subgoal_summaries.append(summary)

    def __reformat_alignments(self, alignments_set, video1, video2):
        for alignment in alignments_set:
            alignment["alignment_title"] = alignment["title"]
            alignment["alignment_description"] = alignment["description"]
            alignment["alignment_reasoning"] = alignment["reasoning"]
            alignment["alignment_comparison"] = alignment["comparison"]
            del alignment["comparison"]
            del alignment["reasoning"]
            del alignment["title"]
            del alignment["description"]

            if "classification" not in alignment:
                alignment["classification"] = ""
            
            if "content_ids" not in alignment:
                alignment["content_ids"] = video1.quotes_to_content_ids(alignment["quotes"])
            if "other_content_ids" not in alignment:
                alignment["other_content_ids"] = video2.quotes_to_content_ids(alignment["other_quotes"])
            alignment["other_video_id"] = video2.video_id
            alignment["id"] = f"link-{video1.video_id}-{random_uid()}"
            
            subgoal_title = alignment["subgoal_title"]

            if len(alignment["other_content_ids"]) == 0:
                ## ASSUMPTION: probably need to show at the end of the subgoal
                if subgoal_title == META_TITLE:
                    alignment["other_content_ids"] = [video2.custom_subgoals[-1]["id"]]
                    continue
                subgoals = video2.get_common_subgoals(subgoal_title)
                for subgoal in reversed(subgoals):
                    if subgoal["title"] == subgoal_title and len(subgoal["content_ids"]) > 0:
                        alignment["other_content_ids"] = [subgoal["content_ids"][-1]]
                        break
        return alignments_set
    
    def __cluster(self, source, items, summarization_f = None):
        if len(items) == 0:
            return []
        prefix = ""
        if source == "alignment" or source == "notable":
            prefix = "alignment"
        else:
            prefix = "notable"
        texts = []
        tags = []
        for item in items:
            texts.append(item[f"{prefix}_description"])
            if source == "alignment":
                ## ASSUMPTION: since comparisons are on the subgoal-level, same cluster should not have alignments from same video&subgoal
                tags.append(item["other_video_id"] + "-" + item["subgoal_title"])
            elif source == "notable":
                ## ASSUMPTION: two different alignments from the same video pair should not be in the same cluster as it should have been eliminated earlier!
                tags.append(item["other_video_id"])
            else:
                ## ASSUMPTION: a hook should not lead to same subgoal of the video twice!
                tags.append(item["video_id"] + "-" + item["subgoal_title"])
        cluster_labels = cluster_tagged_texts(texts, tags)
        # cluster_labels = cluster_texts(texts)
        clusters = {}
        for index, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(items[index])

        if summarization_f is None:
            return list(clusters.values()), []
        
        results = []
        for cluster in clusters.values():
            if len(cluster) < 1:
                continue
            contents = []
            for a in cluster:
                text = (a[f"{prefix}_title"] + ": " + a[f"{prefix}_description"])
                if prefix == "alignment":
                    text += " " + a['alignment_reasoning']
                
                contents.append({
                    "type": "text",
                    "text": text
                })
            if source == "alignment" and len(cluster) == 1:
                results.append({
                    "title": cluster[0][f"{prefix}_title"],
                    "description": cluster[0][f"{prefix}_description"],
                    "reasoning": cluster[0][f"{prefix}_reasoning"],
                    "comparison": cluster[0][f"{prefix}_comparison"],
                })
            else:
                results.append(summarization_f(contents, self.task))
        return results, list(clusters.values())

    ### APPROACH 1
    def __generate_alignments_0(self):
        approach = APPROACHES[0]
        if len(self.videos) < 2:
            return
        if len(self.subgoals) == 0:
            return
        if approach in self.alignment_sets and len(self.alignment_sets[approach]) > 0:
            return
        
        self.alignment_sets[approach] = []
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                cur_alignments = []
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
                        alignment["subgoal_title"] = subgoal_def["title"]

                    cur_alignments.extend(self.__reformat_alignments(subgoal_alignments, video1, video2))
                
                ### between meta
                contents1 = video1.get_meta_summary_contents(False)
                contents2 = video2.get_meta_summary_contents(False)
                meta_alignments = get_meta_alignments_v2(
                    contents1, contents2, self.task
                )
                for alignment in meta_alignments:
                    alignment["subgoal_title"] = META_TITLE
                cur_alignments.extend(self.__reformat_alignments(meta_alignments, video1, video2))

                summarized_alignments, clusters = self.__cluster("alignment", cur_alignments, get_alignments_summary_v2)

                for summarized_alignment, cluster in zip(summarized_alignments, clusters):
                    summarized_alignment["content_ids"] = []
                    summarized_alignment["quotes"] = []
                    summarized_alignment["other_content_ids"] = []
                    summarized_alignment["other_quotes"] = []
                    for alignment in cluster:
                        summarized_alignment["content_ids"] += alignment["content_ids"]
                        summarized_alignment["quotes"] += alignment["quotes"]
                        summarized_alignment["other_content_ids"] += alignment["other_content_ids"]
                        summarized_alignment["other_quotes"] += alignment["other_quotes"]
                    summarized_alignment["subgoal_title"] = cluster[0]["subgoal_title"]

                summarized_alignments = self.__reformat_alignments(summarized_alignments, video1, video2)    
                    
                self.alignment_sets[approach].append({
                    "alignments": summarized_alignments,
                    "video_id": video1.video_id,
                })
    
    ### APPROACH 2
    def __generate_alignments_1(self):
        approach = APPROACHES[1]
        ### TODO: experiment!!!
        if len(self.videos) < 2:
            return
        if len(self.subgoals) == 0:
            return
        if approach in self.alignment_sets and len(self.alignment_sets[approach]) > 0:
            return
        self.alignment_sets[approach] = []
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                cur_alignments = []
                ### between subgoals
                for subgoal_def in self.subgoals:
                    contents1 = video1.get_subgoal_summary_contents(subgoal_def["title"])
                    contents2 = video2.get_subgoal_summary_contents(subgoal_def["title"])
                    if len(contents1) == 0 or len(contents2) == 0:
                        continue
                    subgoal_alignments = get_subgoal_alignments_v3(
                        contents1, contents2, subgoal_def["title"], self.task
                    )
                    for alignment in subgoal_alignments:
                        alignment["subgoal_title"] = subgoal_def["title"]

                    cur_alignments.extend(self.__reformat_alignments(subgoal_alignments, video1, video2))
                
                ### between meta
                contents1 = video1.get_meta_summary_contents(False)
                contents2 = video2.get_meta_summary_contents(False)
                meta_alignments = get_meta_alignments_v3(
                    contents1, contents2, self.task
                )
                for alignment in meta_alignments:
                    alignment["subgoal_title"] = META_TITLE
                cur_alignments.extend(self.__reformat_alignments(meta_alignments, video1, video2))

                ### aggregate alignments
                alignments_per_class = {}
                for alignment in cur_alignments:
                    classification = alignment["classification"]
                    if classification not in alignments_per_class:
                        alignments_per_class[classification] = []
                    alignments_per_class[classification].append(alignment)
                
                summarized_alignments = []
                clusters = []
                for c, a in alignments_per_class.items():
                    cur_summarized, cur_clusters = self.__cluster("alignment", a, get_alignments_summary_v2)
                    for alignment in cur_summarized:
                        alignment["classification"] = c
                    summarized_alignments.extend(cur_summarized)
                    clusters.extend(cur_clusters)

                for summarized_alignment, cluster in zip(summarized_alignments, clusters):
                    summarized_alignment["content_ids"] = []
                    summarized_alignment["quotes"] = []
                    summarized_alignment["other_content_ids"] = []
                    summarized_alignment["other_quotes"] = []
                    for alignment in cluster:
                        summarized_alignment["content_ids"] += alignment["content_ids"]
                        summarized_alignment["quotes"] += alignment["quotes"]
                        summarized_alignment["other_content_ids"] += alignment["other_content_ids"]
                        summarized_alignment["other_quotes"] += alignment["other_quotes"]
                    summarized_alignment["subgoal_title"] = cluster[0]["subgoal_title"]

                summarized_alignments = self.__reformat_alignments(summarized_alignments, video1, video2)    
                    
                self.alignment_sets[approach].append({
                    "alignments": summarized_alignments,
                    "video_id": video1.video_id,
                })
    
    ### APPROACH 3 (BASELINE 1)
    def __generate_alignments_baseline_1(self):
        approach = BASELINES[0]
        ## TODO: UPDATE
        if len(self.videos) < 2:
            return
        if len(self.subgoals) == 0:
            return
        if approach in self.alignment_sets and len(self.alignment_sets[approach]) > 0:
            return
        
        self.alignment_sets[approach] = []
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                cur_alignments = []
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
                    subgoal_alignments = get_subgoal_alignments_v3(
                        context1 + contents1, context2 + contents2, subgoal_def["title"], self.task
                    )
                    for alignment in subgoal_alignments:
                        alignment["subgoal_title"] = subgoal_def["title"]

                    cur_alignments.extend(self.__reformat_alignments(subgoal_alignments, video1, video2))

                ### aggregate alignments
                alignments_per_class = {}
                for alignment in cur_alignments:
                    classification = alignment["classification"]
                    if classification not in alignments_per_class:
                        alignments_per_class[classification] = []
                    alignments_per_class[classification].append(alignment)
                
                summarized_alignments = []
                clusters = []
                for c, a in alignments_per_class.items():
                    cur_summarized, cur_clusters = self.__cluster("alignment", a, get_alignments_summary_v2)
                    for alignment in cur_summarized:
                        alignment["classification"] = c
                    summarized_alignments.extend(cur_summarized)
                    clusters.extend(cur_clusters)

                for summarized_alignment, cluster in zip(summarized_alignments, clusters):
                    summarized_alignment["content_ids"] = []
                    summarized_alignment["quotes"] = []
                    summarized_alignment["other_content_ids"] = []
                    summarized_alignment["other_quotes"] = []
                    for alignment in cluster:
                        summarized_alignment["content_ids"] += alignment["content_ids"]
                        summarized_alignment["quotes"] += alignment["quotes"]
                        summarized_alignment["other_content_ids"] += alignment["other_content_ids"]
                        summarized_alignment["other_quotes"] += alignment["other_quotes"]
                    summarized_alignment["subgoal_title"] = cluster[0]["subgoal_title"]

                summarized_alignments = self.__reformat_alignments(summarized_alignments, video1, video2)    
                    
                self.alignment_sets[approach].append({
                    "alignments": summarized_alignments,
                    "video_id": video1.video_id,
                })
    
    ### APPROACH 4 (BASELINE 2)
    def __generate_alignments_baseline_2(self):
        ## TODO: UPDATE
        approach = BASELINES[1]
        if len(self.videos) < 2:
            return
        if len(self.subgoals) == 0:
            return
        if approach in self.alignment_sets and len(self.alignment_sets[approach]) > 0:
            return
        
        self.alignment_sets[approach] = []
        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                ### between meta
                contents1 = video1.get_all_contents()
                contents2 = video2.get_all_contents()
                if len(contents1) == 0 or len(contents2) == 0:
                    continue
                meta_alignments = get_meta_alignments_v3(
                    contents1, contents2, self.task
                )
                for alignment in meta_alignments:
                    alignment["subgoal_title"] = META_TITLE

                self.alignment_sets[approach].append({
                    "alignments": self.__reformat_alignments(meta_alignments, video1, video2),
                    "video_id": video1.video_id,
                })

    def generate_alignments(self):
        self.__generate_alignments_0()
        self.__generate_alignments_1()
        print("Baseline 1")
        self.__generate_alignments_baseline_1()
        print("Baseline 2")
        self.__generate_alignments_baseline_2()

    def __generate_notable(self, root_alignments):
        if len(root_alignments) < 1:
            return []
        notables = []
        alignments_per_video = {}
        for alignment in root_alignments:
            video_id = alignment["video_id"]
            if video_id not in alignments_per_video:
                alignments_per_video[video_id] = []
            alignments_per_video[video_id] += alignment["alignments"]
        
        for video_id, all_alignments in alignments_per_video.items():
            if len(all_alignments) < 1:
                continue

            video = self.get_video(video_id)
            if video is None:
                continue

            cur_notables, clusters = self.__cluster("notable", all_alignments, get_notable_v2)
            for notable, alignments in zip(cur_notables, clusters):
                if len(alignments) < 1:
                    continue
                ### ASSUMPTION: the smallest content_id & earliest subgoals are the most relevant
                notable_content_id = None
                subgoal_title = None
                for a in alignments:
                    for content_id in a["content_ids"]:
                        if notable_content_id is None or int(notable_content_id.split("-")[-1]) > int(content_id.split("-")[-1]):
                            notable_content_id = content_id
                    subgoal_title = a["subgoal_title"]
                    break
                     
                notables.append({
                    "alignments": alignments,
                    "notable_content_ids": [] if notable_content_id is None else[notable_content_id],
                    "notable_title": notable["title"],
                    "notable_description": notable["description"],
                    "video_id": video_id,
                    "id": f"notable-{video_id}-{random_uid()}",
                    "parent_id": None,
                    "subgoal_title": subgoal_title,
                })
        return notables
    
    def find_notables(self):
        for approach in APPROACHES:
            if approach not in self.alignment_sets:
                continue
            if f"notables_{approach}" not in self.hooks:
                self.hooks[f"notables_{approach}"] = self.__generate_notable(self.alignment_sets[approach])

        for baseline in BASELINES:
            if baseline not in self.alignment_sets:
                continue
            if f"notables_{baseline}" not in self.hooks:
                self.hooks[f"notables_{baseline}"] = self.__generate_notable(self.alignment_sets[baseline])

    def __generate_hooks(self, root_notables):
        hook_parents_per_video = {}
        links_per_video = {}
        for notable in root_notables:
            video_id = notable["video_id"]
            alignments = notable["alignments"]
            notable_title = notable["notable_title"]
            notable_description = notable["notable_description"]
            notable_content_ids = notable["notable_content_ids"]
            notable_id = notable["id"]
            notable_subgoal_title = notable["subgoal_title"]
            for alignment in alignments:
                key = alignment["other_video_id"] + "~" + alignment["classification"]
                if key not in links_per_video:
                    links_per_video[key] = []
                links_per_video[key].append({
                    **alignment,
                    "notable_content_ids": notable_content_ids,
                    "notable_title": notable_title,
                    "notable_description": notable_description,
                    "video_id": video_id,
                    "id": notable_id,
                    "subgoal_title": notable_subgoal_title,
                })

            if video_id not in hook_parents_per_video:
                hook_parents_per_video[video_id] = []
            hook_parents_per_video[video_id].append({
                "id": notable_id,
                "title": notable_title,
                "description": notable_description,
                # "content_ids": notable_content_ids,
                "subgoal_title": notable_subgoal_title,
            })
        
        hooks = []
        for key, links in links_per_video.items():
            video_id = key.split("~")[0]
            classification = key.split("~")[1]
            video = self.get_video(video_id)
            if video is None:
                continue
            hook_parents_embeddings = bert_embedding([hook["description"] for hook in hook_parents_per_video[video_id]])
            
            ### TODO: can try alignment description last sentences!
            cur_hooks, clusters = self.__cluster("hook", links, get_hook_v2)
            
            for hook, cluster in zip(cur_hooks, clusters):
                if (len(cluster) < 1):
                    continue
                
                ### TODO: have to update this part
                ### ASSUMPTION: the most semantiaclly similar content_id is the most relevant
                hook_content_ids = video.get_most_similar_content_ids([hook["description"]])
                
                hook_embedding = bert_embedding([hook["description"]])
                print(hook["description"], hook_content_ids)
                parent_indexes, scores = find_most_similar(hook_parents_embeddings, hook_embedding)
                print(parent_indexes, scores)
                print(hook_parents_per_video[video_id])
                
                parent_id = hook_parents_per_video[video_id][parent_indexes[0]]["id"]
                subgoal_title = hook_parents_per_video[video_id][parent_indexes[0]]["subgoal_title"]
                
                if scores[0] < 0.5:
                    ## ASSUMPTION: If the hook is too dissimilar put it under `undefined`
                    parent_id = None

                hooks.append({
                    "links": cluster,
                    "hook_content_ids": hook_content_ids,
                    "hook_title": hook["title"],
                    "hook_description": hook["description"],
                    "video_id": video_id,
                    "id": f"hook-{video_id}-{random_uid()}",
                    "parent_id": parent_id,
                    "subgoal_title": subgoal_title,
                    "classification": classification,
                })
        return hooks

    def generate_hooks(self):
        for approach in APPROACHES:
            if f"notables_{approach}" not in self.hooks:
                continue
            if f"hooks_{approach}" not in self.hooks:
                self.hooks[f"hooks_{approach}"] = self.__generate_hooks(self.hooks[f"notables_{approach}"])

        for baseline in BASELINES:
            if f"notables_{baseline}" not in self.hooks:
                continue
            if f"hooks_{baseline}" not in self.hooks:
                self.hooks[f"hooks_{baseline}"] = self.__generate_hooks(self.hooks[f"notables_{baseline}"])

    def classify_hooks(self):
        pass

def setup_ds(task_id):
    if task_id not in LIBRARY:
        return None

    # get the video data
    videos = []
    subgoals = []
    video_data_path = f"{PATH}{task_id}/video_data.json"
    subgoal_data_path = f"{PATH}{task_id}/subgoal_data.json"
    alignment_sets = f"{PATH}{task_id}/alignment_sets.json"
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

    if os.path.exists(alignment_sets):
        with open(alignment_sets, "r") as file:
            alignment_sets = json.load(file)
            ds.alignment_sets = alignment_sets
    else:
        ds.alignment_sets = {}

    if os.path.exists(hooks_path):
        with open(hooks_path, "r") as file:
            hooks = json.load(file)
            ds.hooks = hooks
    else:
        ds.hooks = {}

    ds.process_videos()
    
    ds.generate_alignments()

    ds.find_notables()

    ds.generate_hooks()
    
    # ds.classify_alignments()

    export(task_id, ds)
    return ds

def main():

    task_id = list(TASK_DESCRIPTIONS.keys())[0]
    ds = setup_ds(task_id)

    if ds is None:
        return
    
    other_video = '3AAdKl1UYZs'
    current_video = 'D_2DBLAt57c'

    ours_alignments = ds.alignment_sets['approach_1']
    ours2_alignments = ds.alignment_sets['approach_2']
    subgoal_alignments = ds.alignment_sets['baseline_1']
    meta_alignments = ds.alignment_sets['baseline_2']

    for a_set in [ours_alignments, ours2_alignments, subgoal_alignments, meta_alignments]:
        print("## APPROACH")
        for a in a_set:
            if a['video_id'] != current_video:
                continue
            alignments = a["alignments"]
            for alignment in alignments:
                if alignment['other_video_id'] != other_video:
                    continue
                print("-", alignment['alignment_title'])
                print("\t-", alignment['alignment_description'])
                print("\t- class", alignment['classification'])
                print("\t-", alignment['alignment_reasoning'])
                print("\t-", alignment['alignment_comparison'])
                print()
        print()

if __name__ == "__main__":
    main()