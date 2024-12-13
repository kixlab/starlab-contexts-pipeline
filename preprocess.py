import sys
import os
import json

from argparse import ArgumentParser

from helpers import APPROACHES, BASELINES

from src import PATH

from src.Video import Video
from src.VideoPool import VideoPool

def save_data(task_id, pool):
    videos = pool.videos
    taxonomies = pool.taxonomies

    ### save all the video objects
    save_dict = []

    for video in videos:
        save_dict.append(video.to_dict())
    
    if os.path.exists(f"{PATH}{task_id}") is False:
        os.mkdir(f"{PATH}{task_id}")

    with open(f"{PATH}{task_id}/video_data.json", "w") as file:
        json.dump(save_dict, file, indent=2)

    ### save the taxonomies
    with open(f"{PATH}{task_id}/taxonomies.json", "w") as file:
        json.dump(taxonomies, file, indent=2)


def export(task_id, pool):    
    save_data(task_id, pool)

    videos = [video.to_dict(short_metadata=True, fixed_subgoals=True) for video in pool.videos]

    output = {
        "task": pool.task,
        "videos": videos,
        "taxonomies": pool.taxonomies,
    }

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
    taxonomies_path = f"{PATH}{task_id}/taxonomies.json"

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
    
    if os.path.exists(taxonomies_path):
        with open(taxonomies_path, "r") as file:
            taxonomies = json.load(file)
        pool = VideoPool(task_desc, videos, taxonomies)
    else:
        pool = VideoPool(task_desc, videos)

    pool.process_videos()
    
    return pool

def parse_args(args):
    """
    python preprocess.py [-t TASKID]
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--task", dest="task_id", help="Task ID")
    return parser.parse_args(args)

def main(args):
    parsed_args = parse_args(args)
    task_id = parsed_args.task_id
    pool = process_task(task_id)
    export(task_id, pool)

    if pool is None:
        return


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
