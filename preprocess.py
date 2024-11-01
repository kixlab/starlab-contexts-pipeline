import os
import json

from helpers import PATH, LIBRARY, TASK_DESCRIPTIONS, META_TITLE, APPROACHES, BASELINES
from helpers import random_uid

from helpers.Video import Video
from helpers.VideoPool import VideoPool

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
                    for subgoal in video["subgoals"]:
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
            output["hooks"][approach] = []
            if f"hooks_{approach}" in ds.hooks:
                # output["hooks"][approach] += fix_hooks(ds.hooks[f"hooks_{approach}"], "hook")
                output["hooks"][approach] += ds.hooks[f"hooks_{approach}"]
            if f"notables_{approach}" in ds.hooks:
                # output["hooks"][approach] += fix_hooks(ds.hooks[f"notables_{approach}"], "notable")
                output["hooks"][approach] += ds.hooks[f"notables_{approach}"]
            # output["hooks"][approach] += fix_raw_alignments(ds.alignment_sets[approach])

    for baseline in BASELINES:
        if baseline in ds.alignment_sets:
            output["hooks"][baseline] = []
            if f"hooks_{baseline}" in ds.hooks:
                # output["hooks"][baseline] += fix_hooks(ds.hooks[f"hooks_{baseline}"], "hook")
                output["hooks"][baseline] += ds.hooks[f"hooks_{baseline}"]
            if f"notables_{baseline}" in ds.hooks:
                # output["hooks"][baseline] += fix_hooks(ds.hooks[f"notables_{baseline}"], "notable")
                output["hooks"][baseline] += ds.hooks[f"notables_{baseline}"]
            # output["hooks"][baseline] += fix_raw_alignments(ds.alignment_sets[baseline])

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
        ds = VideoPool(task_desc, videos, subgoals)
    else:
        ds = VideoPool(task_desc, videos)

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

    task_id = list(TASK_DESCRIPTIONS.keys())[2]
    ds = setup_ds(task_id)

    if ds is None:
        return
    
    # for video in ds.videos:
    #     contents = video.get_all_contents()
    #     for content in contents:
    #         content["involved"] = []
    #     for aspect in video.meta_summary.keys():
    #         if aspect.endswith("_content_ids"):
    #             for content in contents:
    #                 if (content["id"] in video.meta_summary[aspect]):
    #                     content["involved"].append(aspect)
        
    #     print("####", video.video_id)
    #     for content in contents:
    #         print("-", content["text"])
    #         print("\t-", content["involved"])
    #         print()
    #     print()

    other_video = '3AAdKl1UYZs'
    current_video = 'D_2DBLAt57c'

    # ours_alignments = ds.alignment_sets['approach_1']
    # ours2_alignments = ds.alignment_sets['approach_2']
    # subgoal_alignments = ds.alignment_sets['baseline_1']
    meta_alignments = ds.alignment_sets['baseline_2']

    # for a_set in [ours_alignments, ours2_alignments, subgoal_alignments, meta_alignments]:
    for a_set in [meta_alignments]:
        print("## APPROACH")
        for a in a_set:
            if a['video_id'] != current_video:
                continue
            print ("###", a['video_id'])
            alignments = a["alignments"]
            if len(alignments) == 0:
                continue
            print("###", alignments[0]['other_video_id'])
            for alignment in alignments:
                print("\t-", alignment['alignment_title'])
                print("\t\t-", alignment['alignment_description'])
                print("\t\t- class:", alignment['classification'])
                print("\t\t-", alignment['alignment_reasoning'])
                print("\t\t-", alignment['alignment_comparison'])
                print()
        print()

if __name__ == "__main__":
    main()