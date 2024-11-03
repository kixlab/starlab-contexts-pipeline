import sys
import os
import json

from argparse import ArgumentParser

from helpers import APPROACHES, BASELINES

from src import PATH

from src.Video import Video
from src.VideoPool import VideoPool

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

    ### save all the hooks
    with open(f"{PATH}{task_id}/hooks.json", "w") as file:
        json.dump(hooks, file, indent=2)


def export(task_id, ds):    
    save_data(task_id, ds)

    videos = [video.to_dict(short_metadata=True, fixed_subgoals=True) for video in ds.videos]

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
                output["hooks"][approach] += ds.hooks[f"hooks_{approach}"]
            if f"notables_{approach}" in ds.hooks:
                output["hooks"][approach] += ds.hooks[f"notables_{approach}"]

    for baseline in BASELINES:
        if baseline in ds.alignment_sets:
            output["hooks"][baseline] = []
            if f"hooks_{baseline}" in ds.hooks:
                output["hooks"][baseline] += ds.hooks[f"hooks_{baseline}"]
            if f"notables_{baseline}" in ds.hooks:
                output["hooks"][baseline] += ds.hooks[f"notables_{baseline}"]

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

def process_task(task_id):
    metadata_path = "./metadata.json"
    task_desc = None
    video_pool = None
    with open(metadata_path, "r") as file:
        metadata = json.load(file)
        task_desc = metadata[task_id]["title"]
        video_pool = metadata[task_id]["videos"]


    if video_pool is None:
        raise Exception("No video pool found for task")

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
                if data["video_link"] not in video_pool:
                    continue
                video = Video(data["video_link"])
                video.from_dict(**data)
                videos.append(video)
    if len(videos) == 0 or len(videos[0].subtitles) == 0:
        videos = pre_process_videos(video_pool)

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
    return ds

def parse_args(args):
    """
    python preprocess.py [-t TASKID]
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--task", dest="task_id", help="Task ID")
    return parser.parse_args(args)

def main(args=["-t", "carbonara"]):
    parsed_args = parse_args(args)
    task_id = parsed_args.task_id
    ds = process_task(task_id)
    export(task_id, ds)

    if ds is None:
        return
    
    ## carbonara:
    ## remove-obj: 
    cur_video_id = ""
    other_video_id = ""


if __name__ == "__main__":
    main()
