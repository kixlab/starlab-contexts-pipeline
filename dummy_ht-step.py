# ACTIVITY_NAME = "Shave Chocolate"
# ACTIVITY_NAME = "Make Pastel De Papa" ## too spanish
ACTIVITY_NAME = "Cook Oysters"


import csv
import json
import os

HT_STEP_FOLDER_PATH = "./static/datasets/ht-step"

def get_ht_step(target_activity="Shave Chocolate"):
    """
    Return step taxonomy and videos with segmentations for the given task.
    """
    save_path = os.path.join(HT_STEP_FOLDER_PATH, target_activity.replace(" ", "_").lower() + ".json")
    if os.path.exists(save_path):
        with open(save_path) as f:
            data = json.load(f)
            return data["videos"], data["taxonomy"]


    annotations_path = os.path.join(HT_STEP_FOLDER_PATH, "annotations.json")
    taxonomy_path = os.path.join(HT_STEP_FOLDER_PATH, "taxonomy.csv")
    candidate_steps_path = os.path.join(HT_STEP_FOLDER_PATH, "video_candidate_steps.json")
    
    with open(annotations_path) as f:
        annotations = {}
        annotations = json.load(f)

    with open(taxonomy_path) as f:
        taxonomy = []
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            step = {}
            for i, header in enumerate(headers):
                step[header] = row[i]
            taxonomy.append(step)
    
    with open(candidate_steps_path) as f:
        candidate_steps = {}
        candidate_steps = json.load(f)

    annotated_videos = {}
    found_step_ids = {}
    for video_id in annotations:
        cur_activity = annotations[video_id]["activity"]
        if cur_activity != target_activity:
            continue
        cur_variation = annotations[video_id]["variation"]
        segments = annotations[video_id]["annotations"]
        annotated_videos[video_id] = {
            "activity": cur_activity,
            "variation": cur_variation,
            "segments": segments,
        }
        for segment in segments:
            id = str(segment["id"])
            found_step_ids[id] = True

    step_taxonomy = {}
    for step in taxonomy:
        if step["global_step_index"] in found_step_ids:
            step_taxonomy[step["global_step_index"]] = step

    ### Save the result
    with open(save_path, "w") as f:
        json.dump({
            "videos": annotated_videos,
            "taxonomy": step_taxonomy,
        }, f, indent=2)
    
    return annotated_videos, step_taxonomy


videos_list, step_tax = get_ht_step(ACTIVITY_NAME)

### Download the videos if have not yet
from preprocess import pre_process_videos


def download_videos_ht_step(videos):
    YOUTUBE_PREFIX = "https://www.youtube.com/watch?v="

    video_urls = []
    for video_id in videos_list:
        video_url = YOUTUBE_PREFIX + video_id
        video_urls.append(video_url)

    videos = pre_process_videos(video_urls)
    return videos

video_objs = download_videos_ht_step(videos_list)