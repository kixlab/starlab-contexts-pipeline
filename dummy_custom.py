### Video Processing
FRAMEWORK_PATH = "./static/results/framework/"
MIN_VIDEOS = 20

import json
import os

from preprocess import pre_process_videos

def get_muffin_video_transcripts():
    library_metadata = {}
    with open("./metadata.json") as f:
        library_metadata = json.load(f)

    task_metadata = library_metadata["muffins"]

    muffin_videos = pre_process_videos(task_metadata["videos"])
    
    transcripts = []
    for video in muffin_videos:
        url = f"https://www.youtube.com/watch?v={video.video_id}"
        title = video.metadata["title"]
        content = ""
        transcript = []
        for sentence in video.sentences:
            if sentence['text'].strip() == "":
                continue
            content += f"{sentence['text']}\n"
            transcript.append({
                "text": sentence['text'],
                "start": sentence['start'],
                "end": sentence['finish'],
            })

        transcripts.append({
            "url": url,
            "title": "Making Muffins",
            "original_title": title,
            "content": content,
            "transcript": transcript,
        })
    return transcripts

def get_muffin_articles():
    database_path = "./static/datasets/muffin_articles/"
    articles = []
    
    for filename in os.listdir(database_path):
        with open(database_path + filename) as f:
            ### read line-by-line
            url = f.readline()
            title = f.readline()
            content = ""
            transcript = []
            for idx, line in enumerate(f):
                if line.strip() == "":
                    continue
                content += line
                transcript.append({
                    "text": line.strip(),
                    "start": idx,
                    "end": idx + 1,
                })

            articles.append({
                "url": url,
                "original_title": title,
                "title": "Making Muffins",
                "content": content,
                "transcript": transcript,
            })
    return articles

def get_dataset_muffins(task, dummy=""):
    dataset_filepath = f"{FRAMEWORK_PATH}{task.replace(' ', '_').lower()}_{dummy}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset

    dataset = get_muffin_articles()
    dataset = dataset + get_muffin_video_transcripts()
    print(f"Number of articles: {len(dataset)}")

    # dataset = add_info_labels_to_dataset(dataset, task)

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset

### Handle CrossTask data;
import csv
from helpers.video_scripts import extract_transcript

def library_cross_task():
    library = []
    PATH = "./static/datasets/crosstask/"
    library_path = os.path.join(PATH, "library.json")
    
    if os.path.exists(library_path):
        with open(library_path, "r") as f:
            library = json.load(f)
            return library

    tasks_path = os.path.join(PATH, "crosstask_release/tasks_primary.txt")
    videos_path = os.path.join(PATH, "crosstask_release/videos.csv")
    videos_val_path = os.path.join(PATH, "crosstask_release/videos_val.csv")

    """
    Task ID
    Task name
    URL of corresponding WikiHow page
    Number of steps
    Ordered list of comma-separated steps of the task
    """
    task_obj_ids = ["task_id", "task_name", "url", "num_steps", "steps"]

    with open(tasks_path) as f:
        lines = f.readlines()
        for start_idx in range(0, len(lines), 6):
            cur_task = {}
            finished = False
            for idx, task_obj_id in enumerate(task_obj_ids):
                if start_idx + idx >= len(lines):
                    finished = True
                    break
                cur_task[task_obj_id] = lines[start_idx + idx].strip()
            if finished is False:
                library.append(cur_task)

    for task in library:
        task["steps"] = task["steps"].split(",")
        task["videos"] = []

    for videos_path in [videos_path, videos_val_path]:
        with open(videos_path) as f:
            reader = csv.reader(f)
            for row in reader:
                task_id = row[0]
                video_id = row[1]
                video_url = row[2]
                for task in library:
                    if task["task_id"] == task_id:
                        task["videos"].append({
                            "video_id": video_id,
                            "video_url": video_url,
                        })

    def get_language(video_subtitles_path):
        with open(video_subtitles_path) as f:
            lines = f.readlines()
            for line in lines:
                if "Language:" in line:
                    return line.split(":")[1].strip()
        return None


    SUBTITLES_PATH = os.path.join(PATH, "subtitles")
    for task in library:
        for video in task["videos"]:
            video_id = video["video_id"]
            video_subtitles_path = os.path.join(SUBTITLES_PATH, f"{video_id}.vtt")
            video["subtitles"] = []

            language = get_language(video_subtitles_path)
            if language == "en":
                video["subtitles"] = extract_transcript(video_subtitles_path, None)

    ANNOTATIONS_PATH = os.path.join(PATH, "crosstask_release/annotations/")

    for task in library:
        for video in task["videos"]:
            video["annotations"] = []
            annotation_path = os.path.join(ANNOTATIONS_PATH, f"{task['task_id']}_{video['video_id']}.csv")
            if os.path.exists(annotation_path):
                with open(annotation_path) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        video["annotations"].append({
                            "step": float(row[0]),
                            "start": float(row[1]),
                            "end": float(row[2]),
                        })
            else:
                print(f"No annotation found for {task['task_id']}_{video['video_id']}")

    ### label subtitles with step
    for task in library:
        for video in task["videos"]:
            annotated_subtitles = []
            for subtitle in video["subtitles"]:
                cur_step = None
                for annotation in video["annotations"]:
                    if subtitle["start"] >= annotation["start"] and subtitle["finish"] <= annotation["end"]:
                        cur_step = task["steps"][int(annotation["step"]) - 1]
                        break
                annotated_subtitles.append({
                    **subtitle,
                    "step": cur_step,
                })
            video["subtitles"] = annotated_subtitles

    ### restructure to be similar to the `dataset`

    ### save library as json
    with open(library_path, "w") as f:
        json.dump(library, f, indent=4)


def get_dataset_cross_task(task):
    """
    return dataset with the given task with a structure similar to the `dataset`
    """
    library = library_cross_task()
    dataset = []
    for _task in library:
        if _task["task_name"] == task:
            for video in _task["videos"]:
                content = ""
                transcript = []
                for subtitle in video["subtitles"]:
                    content += f"{subtitle['text']} "
                    transcript.append({
                        "text": subtitle['text'],
                        "start": subtitle['start'],
                        "end": subtitle['finish'],
                    })
                dataset.append({
                    "id": video["video_id"],
                    "url": video["video_url"],
                    "title": task,
                    "original_title": video["task"],
                    "content": content,
                    "transcript": transcript,
                    "steps": [],
                    "ipo": [],
                    "processed_ipos": [],
                })

    ### check if content is enough
    filtered_dataset = []
    for article in dataset:
        if len(article["content"]) < 100:
            continue
        filtered_dataset.append(article)
    dataset = filtered_dataset

    return dataset

def preprocess_cross_task(task, dummy=""):
    dataset_filepath = f"{FRAMEWORK_PATH}{task.replace(' ', '_').lower()}_{dummy}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset


    dataset = get_dataset_cross_task(task)
    print(f"Dataset for {task}: {len(dataset)}")

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset

def preprocess_custom_dataset(task, dummy=""):
    dataset_filepath = f"{FRAMEWORK_PATH}{task.replace(' ', '_').lower()}_{dummy}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset
    
    custom_tasks_path = "./static/datasets/custom-dataset/videos_tasks_per_category.json"
    
    with open(custom_tasks_path) as f:
        custom_tasks = json.load(f)

    videos = []
    category = None
    
    for _category in custom_tasks:
    
        for task_info in custom_tasks[_category]:
            if len(task_info["videos"]) < MIN_VIDEOS:
                continue
            _task = task_info["task_details"]["title"]
    
            if _task == task:
                videos.extend(task_info["videos"])
                category = _category
    

    if category is None:
        raise ValueError(f"Task {task} not found in any category")

    dataset = []
    for video in videos:
        content = ""
        for subtitle in video["transcript"]:
            content += f"{subtitle['text']} "
        dataset.append({
            "id": video["id"],
            "url": "https://www.youtube.com/watch?v=" + video["id"],
            "title": task,
            "original_title": video["title"],
            "category": category,
            "content": content,
            "transcript": video["transcript"],
            "steps": [],
            "ipo": [],
            "processed_ipos": [],
        })

    ### check if content is enough
    filtered_dataset = []
    for article in dataset:
        if len(article["content"]) < 100:
            continue
        filtered_dataset.append(article)
    dataset = filtered_dataset

    print(f"Dataset for {task}: {len(dataset)}")

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset    
### Extracting & Classifying Atmoic Pices Information
import os
import json
from prompts.stupid_experiment_3 import segment_transcript_stupid

TAXONOMY = {
    "opening": "Starting remarks and instructor/channel introductions",
    "closing": "Parting remarks and wrap-up",
    "goal": "Main purpose of the video and its descriptions",
    "motivation": "Reasons or background information on why the video was created",
    "briefing": "Rundown of how the goal will be achieved",
    "subgoal": "Objective of a subsection",
    "instruction": "Actions that the instructor performs to complete the task",
    "tool": "Introduction of the materials, ingredients, and equipment to be used",
    "tip": "Additional instructions or information that makes instructions easier, faster, or more efficient",
    "warning": "Actions that should be avoided",
    "justification": "Reasons why the instruction was performed",
    "effect": "Consequences of the instruction",
    "status": "Descriptions of the current state of the target object",
    "context": "Descriptions of the method or the setting",
    "tool specification": "Descriptions of the tools and equipment",
    "outcome": "Descriptions of the final results of the procedure",
    "reflection": "Summary, evaluation, and suggestions for the future about the overall procedure",
    "side note": "Personal stories, jokes, user engagement, and advertisements",
    "self-promotion": "Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations)",
    "bridge": "Meaningless phrases or expressions that connect different sections",
    "filler": "Conventional filler words",
    "other": "None of the specfied categories",
}

def segment_videos(task, dataset, dummy = ""):
    path = os.path.join(FRAMEWORK_PATH, f'{task.replace(" ", "_").lower()}_segmentation_{dummy}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
        
    for video in dataset:
        video['segmentation'] = segment_transcript_stupid(video['content'], TAXONOMY)

    with open(path, 'w') as f:
        json.dump(dataset, f, indent=4)
    return dataset

### Baseline Processing
from prompts.stupid_experiment_3 import process_transcript_stupid
def process_a_video_baseline_zero_shot(video):
    return process_transcript_stupid(video['title'], video['transcript'])

def process_videos_baseline_zero_shot(task, dataset, dummy=""):
    path = os.path.join(FRAMEWORK_PATH, f'{task.replace(" ", "_").lower()}_zero_shot_baseline_{dummy}.json')
    # if os.path.exists(path):
    #     with open(path) as f:
    #         return json.load(f)
    for video in dataset:
        video['baseline_results'] = process_a_video_baseline_zero_shot(video)
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=4)
    return dataset
### Describing Scope
def describe_scope(video):
    """
    Describes the scope of a video.
    """
    pass

def describe_scope_perplexity(video):
    """
    Describes the scope of a video using LLM perplexity!
    """
    pass
### Main
MUFFIN_TASK = "Making Muffins"

"""
Make French Toast			10 steps / 272 videos
Make Irish Coffee			5 steps / 248 videos
Change a Tire				11 steps / 119 videos
Build (sim.) Floating Shelves		5 steps / 173 videos
"""
CROSS_TASK_TASKS = [
    "Change a Tire",
    "Build (sim.) Floating Shelves",
    "Make French Toast",
    "Make Irish Coffee",
]

CUSTOM_TASKS = [
    ### Food and Entertaining
    "How to Make a Sushi Roll",
    "How to Make Caramel Apples",
    "How to Make a Milkshake Without Ice Cream",
    "How to Grill Steak",
    "How to Make Scrambled Eggs in a Microwave",

    ### Home and Garden
    "How to Grow Hydrangea from Cuttings",
    "How to Grow a Pumpkin",
    "How to Clean Bathroom Tile",
    "How to Polish Stainless Steel",
    "How to Clean a Glass Top Stove",
    "How to Get Rid of a Wasp's Nest",

    # Holidays and Traditions
    "How to Plant a Living Christmas Tree",

    # Sports and Fitness
    "How to Wrap Your Hands for Boxing",
    "How to Catch Trout",

    # Arts and Entertainment
    "How to Make a Paper Hat",
]



def get_dataset(task):
    if task == MUFFIN_TASK:
        return get_dataset_muffins(task, "framework_raw")
    elif task in CROSS_TASK_TASKS:
        return preprocess_cross_task(task, "framework_raw")
    elif task in CUSTOM_TASKS:
        return preprocess_custom_dataset(task, "framework_raw")

def main(task):
    dataset = get_dataset(task)
    process_videos_baseline_zero_shot(task, dataset, dummy="v0")
    # segment_videos(task, dataset, dummy="v0")
task = MUFFIN_TASK
# task = CROSS_TASK_TASKS[0]
# task = CROSS_TASK_TASKS[1]
task = CUSTOM_TASKS[14]

main(task)
