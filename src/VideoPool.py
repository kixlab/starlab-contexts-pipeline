import json

from src import META_TITLE, SIMILARITY_THRESHOLD_NOTABLE, SIMILARITY_THRESHOLD_HOOK

from helpers import APPROACHES, BASELINES
from helpers import random_uid

from helpers.prompts_segmentation import define_steps_v4, extract_subgoals_v4, align_steps_v4, segment_video_v4

from helpers.prompts_summarization import get_step_summary_v4

from helpers.prompts_comparison import get_transcript_alignments_v3
from helpers.prompts_comparison import get_subgoal_alignments_v4, get_steps_alignments_v4

from helpers.prompts_organization import get_notable_v4, get_hooks_v4, get_hook_v4

from helpers.clip import clip_similar_per_text
from helpers.bert import clustering_custom

class VideoPool:
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
        self.__process_videos_v4()

    def __process_videos_v4(self):
        ### Extract steps per video
        for video in self.videos:
            if len(video.steps) == 0:
                video.steps = define_steps_v4(video.get_all_contents(), self.task)

        ### Cluster steps ands define subgoals
        if len(self.subgoals) == 0:
            sequences = []
            for video in self.videos:
                sequences.append(video.steps)
                
            ## ASSUMPTION: as we go over all videos, the steps will get calibrated
            agg_steps = []
            for step in sequences[0]:
                agg_steps.append({
                    "step": step,
                    "original_steps": [step],
                })
            for new_sequence in sequences[1:]:
                old_sequence = []
                for step in agg_steps:
                    old_sequence.append(step["step"])
                
                agg_sequence = align_steps_v4(old_sequence, new_sequence, self.task)
                
                new_agg_steps = []
                for new_agg_step in agg_sequence:
                    original_steps = new_agg_step["original_list_2"]
                    
                    for old_step in new_agg_step["original_list_1"]:
                        for agg_step in agg_steps:
                            if agg_step["step"] == old_step:
                                original_steps += agg_step["original_steps"]
                                break
                    new_agg_steps.append({
                        "step": new_agg_step["aggregated"],
                        "original_steps": original_steps,
                    })
                agg_steps = new_agg_steps
            ### TODO: Turn the sequence of aggregated steps into subgoals
            subgoals = extract_subgoals_v4([s["step"] for s in agg_steps], self.task)
            self.subgoals = []
            for subgoal in subgoals:
                original_steps = []
                for step in subgoal["original_steps"]:
                    for agg_step in agg_steps:
                        if agg_step["step"] == step:
                            original_steps.extend(agg_step["original_steps"])
                            break

                self.subgoals.append({
                    "title": subgoal["title"],
                    "description": subgoal["description"],
                    "original_steps": original_steps,
                })
        
        ### Segment each video based on the appropriate subgoals --> does not work too well...
        for video in self.videos:
            ## initial subgoals
            if len(video.subgoals) == 0:
                segments = segment_video_v4(video.get_all_contents(), video.steps, self.task)
                for index, subgoal in enumerate(segments):
                    video.subgoals.append({
                        "id": f"{video.video_id}-subgoal-{index}",
                        "title": subgoal["title"],
                        "start": subgoal["start"],
                        "finish": subgoal["finish"],
                        "text": subgoal["text"],
                        "frame_paths": subgoal["frame_paths"],
                        "content_ids": subgoal["content_ids"],
                    })
                ## re-segment based on the new subgoals TODO: move under if
                subgoal_assignment = [""] * len(video.subgoals)
                for index, step in enumerate(video.subgoals):
                    for subgoal in self.subgoals:
                        if step["title"] in subgoal["original_steps"]:
                            if subgoal_assignment[index] != "":
                                print(f"Warning: {video.video_id} - {step['title']} is assigned to {subgoal_assignment[index]} and {subgoal['title']}")
                            subgoal_assignment[index] = subgoal["title"]
                new_video_subgoals = []
                for index, step in enumerate(video.subgoals):
                    if len(new_video_subgoals) > 0 and (new_video_subgoals[-1]["title"] == subgoal_assignment[index] or subgoal_assignment[index] == ""):
                        new_video_subgoals[-1]["finish"] = step["finish"]
                        new_video_subgoals[-1]["text"] += " " + step["text"]
                        new_video_subgoals[-1]["frame_paths"] += step["frame_paths"]
                        new_video_subgoals[-1]["content_ids"] += step["content_ids"]
                        new_video_subgoals[-1]["original_steps"].append(step["title"])
                    else:
                        new_video_subgoals.append({
                            "id": step["id"],
                            "title": subgoal_assignment[index],
                            "start": step["start"],
                            "finish": step["finish"],
                            "text": step["text"],
                            "frame_paths": step["frame_paths"],
                            "content_ids": step["content_ids"],
                            "original_steps": [step["title"]],
                        })
                video.subgoals = new_video_subgoals

            ### extract useful information for each step
            if len(video.subgoal_summaries) == 0:
                video.subgoal_summaries = []
                for subgoal in video.subgoals:
                    if subgoal["title"] == "":
                        continue
                    summary = get_step_summary_v4(video.get_all_contents(),
                        subgoal["original_steps"], self.task)
                    for key in summary:
                        if not key.endswith("_content_ids"):
                            continue
                        new_content_ids = []
                        for id in summary[key]:
                            new_content_ids.append(f"{video.video_id}-{id}")
                        summary[key] = new_content_ids
                    summary = {
                        "title": subgoal["title"],
                        **summary,
                        "outcome_frame_paths": [],
                        "materials_frame_paths": [],
                        "tools_frame_paths": [],
                    }
                    if len(summary["outcome"]) > 0:
                        summary["outcome_frame_paths"] = clip_similar_per_text([*summary["outcome"]], summary["frame_paths"])
                    if len(summary["materials"]) > 0:
                        summary["materials_frame_paths"] = clip_similar_per_text([*summary["materials"]], summary["frame_paths"])
                    if len(summary["tools"]) > 0:
                        summary["tools_frame_paths"] = clip_similar_per_text([*summary["tools"]], summary["frame_paths"])
                    video.subgoal_summaries.append(summary)

            cannot_be_empty = ["instructions", "explanation", "tips"]
            for subgoal_summary in video.subgoal_summaries:
                for key in cannot_be_empty:
                    if len(subgoal_summary[key+"_content_ids"]) == 0:
                        subgoal_summary[key] = ""

    def __reformat_alignments_v2(self, alignments, video1, video2):
        for alignment in alignments:
            alignment["other_video_id"] = video2.video_id
            alignment["id"] = f"link-{video1.video_id}-{random_uid()}"
            alignment["alignment_title"] = alignment["title"]
            alignment["alignment_description"] = alignment["description"]
            alignment["alignment_reasoning"] = alignment["reasoning"]
            alignment["alignment_comparison"] = alignment["comparison"]
            alignment["seconds"] = video1.get_alignment_seconds(alignment)
            del alignment["comparison"]
            del alignment["reasoning"]
            del alignment["title"]
            del alignment["description"]
        return alignments

    ### APPROACH 1 (CLASSIFICATION + IMPORTANCE + AGGREGATION_PER_RELATION_AND_CLASS)
    def __generate_alignments_1(self):
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
                # TODO: check one-by-one generation!
                if v1_idx >= v2_idx:
                    continue
                alignments_1 = []
                alignments_2 = []
                ### between subgoals
                for subgoal_def in self.subgoals:
                    contents1 = video1.get_subgoal_summary_multimodal_contents(subgoal_def["title"])
                    contents2 = video2.get_subgoal_summary_multimodal_contents(subgoal_def["title"])
                    if len(contents1) == 0 or len(contents2) == 0:
                        continue
                    subgoal_alignments_1, subgoal_alignments_2 = get_subgoal_alignments_v4(
                        contents1, contents2, subgoal_def["title"], self.task
                    )
                    for alignment in [*subgoal_alignments_1, *subgoal_alignments_2]:
                        alignment["subgoal_title"] = subgoal_def["title"]
                    alignments_1.extend(self.__reformat_alignments_v2(
                        subgoal_alignments_1, video1, video2
                    ))
                    alignments_2.extend(self.__reformat_alignments_v2(
                        subgoal_alignments_2, video2, video1
                    ))
                
                ### between meta
                meta_alignments_1, meta_alignments_2 = get_steps_alignments_v4(
                    contents1, contents2, self.task
                )
                for alignment in [*meta_alignments_1, *meta_alignments_2]:
                    alignment["subgoal_title"] = META_TITLE
                
                alignments_1.extend(self.__reformat_alignments_v2(
                    meta_alignments_1, video1, video2
                ))
                alignments_2.extend(self.__reformat_alignments_v2(
                    meta_alignments_2, video2, video1
                ))
                    
                self.alignment_sets[approach].append({
                    "alignments": alignments_1,
                    "video_id": video1.video_id,
                })
                self.alignment_sets[approach].append({
                    "alignments": alignments_2,
                    "video_id": video2.video_id,
                })
    
    ### BASELINE 1
    def __generate_alignments_baseline_1(self):
        approach = BASELINES[0]
        if len(self.videos) < 2:
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
                meta_alignments = get_transcript_alignments_v3(
                    contents1, contents2, self.task
                )
                for alignment in meta_alignments:
                    alignment["subgoal_title"] = META_TITLE

                self.alignment_sets[approach].append({
                    "alignments": self.__reformat_alignments(meta_alignments, video1, video2),
                    "video_id": video1.video_id,
                })

    def generate_alignments(self):
        self.__generate_alignments_1()
        # self.__generate_alignments_baseline_1()

    def __cluster_v2(self, items, similarity_threshold, item_to_text_f, summarization_f):
        if len(items) == 0:
            return []
        texts = []
        for item in items:
            texts.append(item_to_text_f(item))
        cluster_labels = clustering_custom(texts, similarity_threshold)
        clusters = {}
        for index, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(index)
        
        results = []
        for cluster in clusters.values():
            if len(cluster) < 1:
                continue
            cur_items = [items[index] for index in cluster]
            summary = summarization_f(cur_items)
            results.append({
                **summary,
                "links": cluster,
            })
        return results

    def __generate_notable_v2(self, root_alignments):
        if len(root_alignments) < 1:
            return []
        notables = []
        alignments_per_video = {}
        for alignment in root_alignments:
            video_id = alignment["video_id"]
            if video_id not in alignments_per_video:
                alignments_per_video[video_id] = []
            alignments_per_video[video_id] += alignment["alignments"]

        def __calculate_notable_importance(links):
            ## Average of importance of each link
            if len(links) < 1:
                return 0
            importance = 0
            for link in links:
                importance += link["importance"]
            return importance / len(links)
        
        def __calculate_notable_seconds(links):
            seconds = 0
            for link in links:
                seconds = max(seconds, link["seconds"])
            return seconds
        
        def __get_notable_links_contents(links):
            contents = []
            for link in links:
                text = "- **Procedural Content**: " + link["title"] + "\n"
                text += "\t- Content Description: " + link["description"] + "\n"
                text += "\t- Reasoning: " + link['reasoning'] + "\n"
                text += "\t- Comparison to Other Tutorials: " + link['comparison'] + "\n"    
                contents.append({
                    "type": "text",
                    "text": text
                })
            return contents
        
        def __link_to_text(link):
            return link["title"] + ": " + link["description"]

        for video_id, all_alignments in alignments_per_video.items():
            if len(all_alignments) < 1:
                continue

            video = self.get_video(video_id)
            if video is None:
                continue

            alignments_per_subgoal_aspect = {}
            for alignment in all_alignments:
                subgoal_aspect = alignment["subgoal_title"] + "+" + alignment["aspect"]

                if subgoal_aspect not in alignments_per_subgoal_aspect:
                    alignments_per_subgoal_aspect[subgoal_aspect] = []
                alignments_per_subgoal_aspect[subgoal_aspect].append({
                    "id": alignment["id"],
                    "other_video_id": alignment["other_video_id"],
                    "title": alignment["alignment_title"],
                    "description": alignment["alignment_description"],
                    "reasoning": alignment["alignment_reasoning"],
                    "comparison": alignment["alignment_comparison"],
                    "aspect": alignment["aspect"],
                    "subgoal": alignment["subgoal_title"],
                    "relation": alignment["relation"],
                    "importance": alignment["importance"],
                    "seconds": alignment["seconds"],
                })

            ### cluster alignments per aspect
            for subgoal_aspect, alignments in alignments_per_subgoal_aspect.items():
                new_notables = []
                if len(alignments) < 1:
                    continue
                subgoal = subgoal_aspect.split("+")[0]
                aspect = subgoal_aspect.split("+")[1]
                ### TODO: costly but can try llm
                ### clustering
                def __get_notable(links):
                    if len(links) < 1:
                        return None

                    if len(links) == 1:
                        return {
                            "title": links[0]["title"],
                            "description": links[0]["description"],
                            "reasoning": links[0]["reasoning"],
                            "comparison": links[0]["comparison"],
                        }

                    contents = __get_notable_links_contents(links)
                    summary = get_notable_v4(contents, subgoal, aspect, self.task)
                    return summary
                new_notables = self.__cluster_v2(alignments, SIMILARITY_THRESHOLD_NOTABLE, __link_to_text, __get_notable)

                for notable in new_notables:
                    cur_links = [alignments[index] for index in notable["links"]]
                    ### merge links with the same video_id
                    cur_links_dict = {}
                    for link in cur_links:
                        key = link["other_video_id"] + "+" + link["relation"]
                        if key not in cur_links_dict:
                            cur_links_dict[key] = []
                        cur_links_dict[key].append(link)
                    merged_links = []
                    for key, links in cur_links_dict.items():
                        other_video_id = key.split('+')[0]
                        relation = key.split('+')[1]
                        new_link = __get_notable(links)
                        merged_links.append({
                            "id": links[0]["id"],
                            "other_video_id": other_video_id,
                            "title": new_link["title"],
                            "description": new_link["description"],
                            "reasoning": new_link["reasoning"],
                            "comparison": new_link["comparison"],
                            "aspect": aspect,
                            "subgoal": subgoal,
                            "relation": relation,
                            "importance": __calculate_notable_importance(links),
                            "seconds": __calculate_notable_seconds(links),
                        })

                    notables.append({
                        "id": f"notable-{video_id}-{random_uid()}",
                        "video_id": video_id,
                        "title": notable["title"],
                        "description": notable["description"],
                        "reasoning": notable["reasoning"],
                        "comparison": notable["comparison"],
                        "subgoal": subgoal,
                        "aspect": aspect,
                        "links": merged_links,
                        "importance": __calculate_notable_importance(cur_links),
                        "step_aspect_complexity": len(new_notables),
                        "uniqueness": 0,
                        "seconds": __calculate_notable_seconds(cur_links),
                    })
        
        ### sort links of notables by importance
        for notable in notables:
            notable["links"] = sorted(notable["links"], key=lambda x: x["importance"], reverse=True)

        ### calculate uniquness of each notable
        video_cnt = len(alignments_per_video.keys())
        if video_cnt < 1:
            video_cnt = 1
        for notable in notables:
            notable["uniqueness"] = len(notable["links"]) / video_cnt
        return notables
    
    def find_notables(self):
        for approach in APPROACHES:
            if approach not in self.alignment_sets:
                continue
            if f"notables_{approach}" not in self.hooks:
                self.hooks[f"notables_{approach}"] = self.__generate_notable_v2(self.alignment_sets[approach])

        for baseline in BASELINES:
            if baseline not in self.alignment_sets:
                continue
            if f"notables_{baseline}" not in self.hooks:
                self.hooks[f"notables_{baseline}"] = self.__generate_notable_v2(self.alignment_sets[baseline])

    def __generate_hooks_v2(self, root_notables, approach="cluster"): #llm
        links_to = {}
        for notable in root_notables:
            for link in notable["links"]:
                key = link["other_video_id"] + "+" + link["subgoal"] + "+" + link["relation"] + "+" + link["aspect"]
                if key not in links_to:
                    links_to[key] = []
                links_to[key].append({
                    "id": notable["id"],
                    "title": notable["title"],
                    "description": notable["description"],
                    "reasoning": link["reasoning"],
                    "comparison": link["comparison"],
                    "subgoal": notable["subgoal"],
                    "aspect": notable["aspect"],
                    "relation": link["relation"],
                    "other_video_id": notable["video_id"],
                    "importance": notable["importance"],
                    "uniqueness": notable["uniqueness"],
                    "step_aspect_complexity": notable["step_aspect_complexity"],
                    "other_seconds": notable["seconds"],
                })

        def __calculate_hook_importance(links):
            ## Max Link
            importance = 0
            for link in links:
                importance = max(importance, link["importance"])
            return importance
    
        def __get_hook_links_contents(links):
            contents = []
            for link in links:
                text = "- **Procedural Content**: " + link["title"] + "\n"
                text += "\t- Content Description: " + link["description"] + "\n"
                # text += "\t- Reasoning: " + link['reasoning'] + "\n"
                text += "\t- Comparison to Current Tutorial: " + link['comparison'] + "\n"  
                contents.append({
                    "type": "text",
                    "text": text
                })
            return contents
        
        def __link_to_text(link):
            return link["comparison"]
        
        all_hooks = []
        for key, links in links_to.items():
            video_id = key.split("+")[0]
            subgoal = key.split("+")[1]
            relation = key.split("+")[2]
            aspect = key.split("+")[3]

            video = self.get_video(video_id)
            if video is None or len(links) < 1:
                continue
            new_hooks = []
            if approach == "llm":
                contents = __get_hook_links_contents(links)
                new_hooks = get_hooks_v4(contents, subgoal, relation, aspect, self.task)
            else:
                ### clustering
                def __get_hook(links):
                    if len(links) < 1:
                        return None

                    if len(links) == 1:
                        return {
                            "title": relation.capitalize() + " " + aspect.capitalize(),
                            "description": links[0]["description"],
                            "comparison": links[0]["comparison"],
                        }
                    contents = __get_hook_links_contents(links)
                    summary = get_hook_v4(contents, subgoal, relation, aspect, self.task)
                    return summary
                new_hooks = self.__cluster_v2(links, SIMILARITY_THRESHOLD_HOOK, __link_to_text, __get_hook)

            # combine hooks with the same title
            new_hooks_dict = {}
            for hook in new_hooks:
                if hook["title"] not in new_hooks_dict:
                    new_hooks_dict[hook["title"]] = []
                new_hooks_dict[hook["title"]].append(hook)
            
            merged_hooks = []
            for title, hooks_per_title in new_hooks_dict.items():
                if len(hooks_per_title) == 1:
                    merged_hooks.append(hooks_per_title[0])
                else:
                    descriptions = [hook["description"] for hook in hooks_per_title]
                    comparisons = [hook["comparison"] for hook in hooks_per_title]
                    cur_links = [index for hook in hooks_per_title for index in hook["links"]]
                    cur_links = list(set(cur_links))
                    merged_hooks.append({
                        "title": title,
                        "description": " ".join(descriptions),
                        "comparison": " ".join(comparisons),
                        "links": cur_links,
                    })

            for hook in merged_hooks:
                cur_links = [links[index] for index in hook["links"]]
                all_hooks.append({
                    "id": f"hook-{video_id}-{random_uid()}",
                    "video_id": video_id,
                    "subgoal": subgoal,
                    "aspect": aspect,
                    "relation": relation,
                    "title": hook["title"],
                    "description": hook["description"],
                    "comparison": hook["comparison"],
                    "links": cur_links,
                    "importance": __calculate_hook_importance(cur_links),
                })
        
        ### sort links of hook by importance
        for hook in all_hooks:
            hook["links"] = sorted(hook["links"], key=lambda x: x["importance"], reverse=True)
        return all_hooks

    def generate_hooks(self):
        for approach in APPROACHES:
            if f"notables_{approach}" not in self.hooks:
                continue
            if f"hooks_{approach}" not in self.hooks:
                self.hooks[f"hooks_{approach}"] = self.__generate_hooks_v2(self.hooks[f"notables_{approach}"])

        for baseline in BASELINES:
            if f"notables_{baseline}" not in self.hooks:
                continue
            if f"hooks_{baseline}" not in self.hooks:
                self.hooks[f"hooks_{baseline}"] = self.__generate_hooks_v2(self.hooks[f"notables_{baseline}"])