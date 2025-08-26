from helpers.bert import hierarchical_clustering
from prompts.stupid_experiment_2 import aggregate_steps_stupid

from prompts.stupid_experiment_2 import extract_steps

def add_steps_to_dataset(dataset, task):

    for article in dataset:
        tutorial = article["title"] + "\n" + article["content"]
        steps = extract_steps(task, tutorial)
        article["steps"] = steps

    return dataset

def aggregate_hierarchical(items, task, distance_threshold=0.2):

    clusters = hierarchical_clustering(items, embedding_method="bert", linkage="average", n_clusters=None, distance_threshold=distance_threshold)

    # mappings = {}
    # for i, cluster in enumerate(clusters):
    #     if cluster not in mappings:
    #         mappings[cluster] = []
    #     mappings[cluster].append(items[i])

    # print(json.dumps(mappings, indent=4))
    return clusters


def construct_step_taxonomy(dataset, task, dummy=""):
    step_taxonomy_filepath = f"./static/results/{task.replace(' ', '_').lower()}_taxonomy_{dummy}.json"
    if os.path.exists(step_taxonomy_filepath):
        with open(step_taxonomy_filepath) as f:
            taxonomy = json.load(f)
        return taxonomy
    
    all_steps = []
    for article in dataset:
        for step in article["steps"]:
            all_steps.append({
                "step": step["step"],
                "description": step["description"],
                "original_tutorial": article["url"],
            })
    ### hierarchical clustering
    # taxonomy = aggregate_hierarchical(all_steps, task)

    ### LLM-based stupid aggregation
    taxonomy = aggregate_steps_stupid(task, all_steps)

    with open(step_taxonomy_filepath, "w") as f:
        json.dump(taxonomy, f, indent=4)
    return taxonomy

from prompts.stupid_experiment_2 import extract_ipos

def extract_ipos_stupid(dataset, taxonomy, task, dummy=""):
    """
    TODO: can potentially simplify the input-output-instruction extraction, since we are taxonomizing anyway...
    """
    ipos_filepath = f"./static/results/{task.replace(' ', '_').lower()}_ipos_{dummy}.json"
    if os.path.exists(ipos_filepath):
        with open(ipos_filepath) as f:
            dataset = json.load(f)
        return dataset
    
    for article in dataset:
        tutorial = article["title"] + "\n" + article["content"]
        article["ipo"] = extract_ipos(task, taxonomy, tutorial)
    
    with open(ipos_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset
    
from prompts.stupid_experiment_2 import taxonomize_ipos_stupid
IPO_KEYS = ["inputs", "outputs", "methods"]

def taxonomize_ipos(dataset, taxonomy, task, dummy=""):
    ipo_taxonomy_filepath = f"./static/results/{task.replace(' ', '_').lower()}_ipo_taxonomy_{dummy}.json"
    if os.path.exists(ipo_taxonomy_filepath):
        with open(ipo_taxonomy_filepath) as f:
            ipo_taxonomy = json.load(f)
        return ipo_taxonomy

    ipo_per_step = {}
    for step in taxonomy:
        ipo_per_step[step["step"]] = {
            "phase": step["phase"],
            "tutorials": [],
        }

    for article in dataset:
        for step_info in article["ipo"]:
            step = step_info["step"]
            if step not in ipo_per_step:
                print(f"Error: Step {step} not found in taxonomy")
                continue
            cur_entry = {}
            for ipo_key in IPO_KEYS:
                if ipo_key not in step_info:
                    print(f"Error: {ipo_key} not found in step {step}")
                    continue
                cur_entry[ipo_key] = step_info[ipo_key]
            ipo_per_step[step]["tutorials"].append(cur_entry)

    ipo_taxonomy = {}

    for step in ipo_per_step:
        subtask = ipo_per_step[step]["phase"] + ": " + step
        tutorials = ipo_per_step[step]["tutorials"]
        ipo_taxonomy[step] = taxonomize_ipos_stupid(task, tutorials, subtask)

    with open(ipo_taxonomy_filepath, "w") as f:
        json.dump(ipo_taxonomy, f, indent=4)
    return ipo_taxonomy


    ### TODO: Try each component separately
    # for article in dataset:
    #     for step_info in article["ipo"]:
    #         step = step_info["step"]
    #         if step not in ipo_per_step:
    #             print(f"Error: Step {step} not found in taxonomy")
    #             continue
    #         for ipo_key in IPO_KEYS:
    #             if ipo_key not in step_info:
    #                 print(f"Error: {ipo_key} not found in step {step}")
    #                 continue
    #             if ipo_key not in ipo_per_step[step]:
    #                 ipo_per_step[step][ipo_key] = []
    #             ipo_per_step[step][ipo_key].append({
    #                 "present": step_info["present"],
    #                 "set": step_info[ipo_key],
    #                 "original_tutorial": article["url"],
    #             })
    
    # ipo_taxonomy_per_step = {}

    # for step in ipo_per_step:
    #     if step not in ipo_taxonomy_per_step:
    #         ipo_taxonomy_per_step[step] = {
    #             "phase": ipo_per_step[step]["phase"],
    #         }

    #     # for ipo_key in IPO_KEYS:
    #     #     if ipo_key not in ipo_taxonomy_per_step[step]:
    #     #         ipo_taxonomy_per_step[step][ipo_key] = []
    #     #     if ipo_key not in ipo_per_step[step]:
    #     #         ipo_per_step[step][ipo_key] = []
    #     #         continue
    #     #     ### taxonomize the sets: enter the entire lists with some context and try to aggregate wrt other parts of the IPO

from prompts.stupid_experiment_2 import extract_information_per_ipo_stupid

def extract_information_per_ipo(dataset, step_taxonomy, ipo_taxonomy, task, dummy=""):
    dataset_filepath = f"./static/results/{task.replace(' ', '_').lower()}_ipo_information_{dummy}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset
    
    step_to_subtask = {}
    for step in step_taxonomy:
        step_to_subtask[step["step"]] = f"{step['phase']}: {step['step']}"

    async_calls = []

    for article in dataset:
        article["ipo_information"] = []
        for step_info in article["ipo"]:
            step = step_info["step"]
            if step not in ipo_taxonomy:
                print(f"Error: Step {step} not found in ipo_taxonomy")
                continue
            if step not in step_to_subtask:
                print(f"Error: Step {step} not found in step_to_subtask")
                continue
            tutorial = article["title"] + "\n" + article["content"]
            subtask = step_to_subtask[step]
            step_ipo_taxonomy = ipo_taxonomy[step]
            cur_information = extract_information_per_ipo_stupid(task, step_ipo_taxonomy, tutorial, subtask)
            article["ipo_information"].append({
                "step": step,
                ### ideally add the phase?
                "subtask": subtask,
                **cur_information,
            })
            # async_calls.append((task, step_ipo_taxonomy, tutorial, subtask))

    # print(f"Number of async calls: {len(async_calls)}")
    # async def process_call(task, step_ipo_taxonomy, tutorial, subtask):
    #     return await extract_information_per_ipo_stupid(task, step_ipo_taxonomy, tutorial, subtask)

    # loop = asyncio.get_event_loop()
    # results = await asyncio.gather(*[process_call(task, step_ipo_taxonomy, tutorial, subtask) for task, step_ipo_taxonomy, tutorial, subtask in async_calls])
    # for i, article in enumerate(dataset):
    #     for step_info in article["ipo"]:
    #         step = step_info["step"]
    #         if step not in ipo_taxonomy:
    #             print(f"Error: Step {step} not found in ipo_taxonomy")
    #             continue
    #         if step not in step_to_subtask:
    #             print(f"Error: Step {step} not found in step_to_subtask")
    #             continue
    #         subtask = step_to_subtask[step]
    #         cur_information = results[i]
    #         article["ipo_information"].append({
    #             "step": step,
    #             ### ideally add the phase?
    #             "subtask": subtask,
    #             **cur_information,
    #         })

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset


# CrossTask
### Representation construction
import os
import csv
import json
from helpers.video_scripts import extract_transcript

def library_cross_task():
    library = []
    PATH = "./static/datasets/crosstask/"
    library_path = os.path.join(PATH, "library.json")
    
    if os.path.exists(library_path):
        with open(library_path, "r") as f:
            library = json.load(f)
            return library

    TASKS_PATH = os.path.join(PATH, "crosstask_release/tasks_primary.txt")
    VIDEOS_PATH = os.path.join(PATH, "crosstask_release/videos.csv")
    VIDEOS_VAL_PATH = os.path.join(PATH, "crosstask_release/videos_val.csv")

    """
    Task ID
    Task name
    URL of corresponding WikiHow page
    Number of steps
    Ordered list of comma-separated steps of the task
    """
    task_obj_ids = ["task_id", "task_name", "url", "num_steps", "steps"]

    with open(TASKS_PATH) as f:
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

    for videos_path in [VIDEOS_PATH, VIDEOS_VAL_PATH]:
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
                for subtitle in video["subtitles"]:
                    content += f"{subtitle['text']}\n"
                dataset.append({
                    "id": video["video_id"],
                    "url": video["video_url"],
                    "title": task,
                    "content": content,
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

    ### remove duplicate urls
    filtered_dataset = {}
    for article in dataset:
        if article["url"] in filtered_dataset:
            continue
        filtered_dataset[article["url"]] = article
    dataset = list(filtered_dataset.values())

    return dataset

### settings
# INFORMATION_KEYS = ["description", "explanation", "tips", "alternatives"]
INFORMATION_KEYS = ["description", "instruction", "explanations", "tips"]
# AGGREGATION_APPROACH = "clustering"
AGGREGATION_APPROACH = "llm-based"
INFORMATION_AGGREGATION_DISTANCE_THRESHOLD = 0.5

KEY_LEVELS = {
    "r": 0,
    "phase": 1,
    "step": 2,
    "ipo": 3,
    "element": 4,
}
from prompts.stupid_experiment_2 import cluster_information_stupid

def extract_task_representation(dataset, step_taxonomy, ipo_taxonomy, task, dummy=""):
    representation_filepath = f"./static/results/{task.replace(' ', '_').lower()}_representation_{dummy}.json"

    if os.path.exists(representation_filepath):
        with open(representation_filepath) as f:
            representation = json.load(f)
        return representation
    
    step_to_phase = {}
    for step in step_taxonomy:
        step_to_phase[step["step"]] = step["phase"]

    ### covered articles; todo: remove later
    covered_articles = {}

    representation = {} # phase&step --> ipo_taxonomy --> elements --> information

    ### extract the task representation
    for article in dataset:
        if article["url"] in covered_articles:
            continue
        covered_articles[article["url"]] = 1
        for step_info in article["ipo_information"]:
            step = step_info["step"]
            phase = step_to_phase[step]
            if phase not in representation:
                representation[phase] = {}
            if step not in representation[phase]:
                representation[phase][step] = {}
            for ipo_key in IPO_KEYS:
                if ipo_key not in step_info:
                    continue
                if ipo_key not in representation[phase][step]:
                    representation[phase][step][ipo_key] = {}
                for element in step_info[ipo_key]:
                    element_name = element["name"]
                    is_present = element["present"]
                    if is_present is False:
                        continue
                    if element_name not in representation[phase][step][ipo_key]:
                        representation[phase][step][ipo_key][element_name] = []
                    for information_key in INFORMATION_KEYS:
                        if information_key not in element:
                            continue
                        if isinstance(element[information_key], list):
                            for text in element[information_key]:
                                representation[phase][step][ipo_key][element_name].append({
                                    "url": article["url"],
                                    "content": text,
                                    "type": information_key
                                })
                        else:
                            representation[phase][step][ipo_key][element_name].append({
                                "url": article["url"],
                                "content": element[information_key],
                                "type": information_key,
                            })
                    
    def aggregate_information(items, key, context):
        contents = []
        for item in items:
            contents.append(item["content"])
        if AGGREGATION_APPROACH == "clustering":
            clusters = aggregate_hierarchical(contents, context, distance_threshold=INFORMATION_AGGREGATION_DISTANCE_THRESHOLD)
        elif AGGREGATION_APPROACH == "llm-based":
            clusters = cluster_information_stupid(task, contents, context, key)

        result = []
        information_taxonomy = {}
        for idx, cluster in enumerate(clusters):
            if cluster not in information_taxonomy:
                information_taxonomy[cluster] = []
            information_taxonomy[cluster].append(items[idx])

        for cluster in information_taxonomy:
            cur_items = information_taxonomy[cluster]
            rep = {
                "type": key,
                "content": cur_items[0]["content"],
                "items": cur_items
            }
            result.append(rep)
        return result


    ### cluster similar element information
    for phase in representation:
        for step in representation[phase]:
            for ipo_key in representation[phase][step]:
                for element in representation[phase][step][ipo_key]:
                    items = representation[phase][step][ipo_key][element]
                    ### cluster the descriptions
                    ### TODO: can do this across each level for each type of info and see the overlap???
                    context = f"{phase} -> {step} -> {ipo_key} -> {element}"
                    library = []
                    for information_key in INFORMATION_KEYS:
                        ### some are str and some are list[str]
                        actual_items = []
                        for item in items:
                            if item["type"] == information_key:
                                actual_items.append({
                                    "url": item["url"],
                                    "content": item["content"]
                                })
                        library.extend(
                            aggregate_information(actual_items, information_key, context)
                        )
                    representation[phase][step][ipo_key][element] = library
                        
    ### save the representation
    with open(representation_filepath, "w") as f:
        json.dump(representation, f, indent=4)
    return representation

def preprocess_cross_task(task, dummy=""):
    dataset_filepath = f"./static/results/{task.replace(' ', '_').lower()}_{dummy}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset


    dataset = get_dataset_cross_task(task)
    print(f"Dataset for {task}: {len(dataset)}")
    dataset = add_steps_to_dataset(dataset, task)

    with open(dataset_filepath, "w") as f:
        print(f"DUMPED {task} dataset to {dataset_filepath}")
        json.dump(dataset, f, indent=4)
    return dataset

def cross_task(task_idx):
    """
    Make French Toast			10 steps / 272 videos
    Make Irish Coffee			5 steps / 248 videos
    Change a Tire				11 steps / 119 videos
    Build (sim.) Floating Shelves		5 steps / 173 videos

    """

    CROSS_TASK_TASKS = [
        "Make French Toast",
        "Make Irish Coffee",
        "Change a Tire",
        "Build Simple Floating Shelves",
    ]

    task = CROSS_TASK_TASKS[task_idx]
    dataset = preprocess_cross_task(task, "raw")
    taxonomy = construct_step_taxonomy(dataset, task, "stupid")
    dataset = extract_ipos_stupid(dataset, taxonomy, task, "stupid")
    ipo_taxonomy = taxonomize_ipos(dataset, taxonomy, task, "stupid")
    dataset = extract_information_per_ipo(dataset, taxonomy, ipo_taxonomy, task, "stupid")
    print("pre-process done!")

    representation = extract_task_representation(dataset, taxonomy, ipo_taxonomy, task, "stupid-" + AGGREGATION_APPROACH)
    tutorial_urls = []
    url_to_content = {}
    url_to_nice_name = {}
    for idx, article in enumerate(dataset):
        url_to_content[article["url"]] = article["content"]
        tutorial_urls.append(article["url"])
        url_to_nice_name[article["url"]] = f"T{idx}"

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
        for sentence in video.sentences:
            content += f"{sentence['text']}\n"

        transcripts.append({
            "url": url,
            "title": title,
            "content": content
        })
    return transcripts

def get_muffin_articles():
    PATH = "./static/database/"
    PREFIX = "muffin_articles_"
    SUFFIX = ".txt"
    articles = []
    
    for filename in os.listdir(PATH):
        if filename.startswith(PREFIX) and filename.endswith(SUFFIX):
            with open(PATH + filename) as f:
                ### read line-by-line
                url = f.readline()
                title = f.readline()
                content = ""
                for line in f:
                    content += line

                articles.append({
                    "url": url,
                    "title": title,
                    "content": content
                })
    return articles

def get_dataset_muffins(task, dummy=""):
    dataset_filepath = f"./static/results/{task.replace(' ', '_').lower()}_{dummy}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset

    dataset = get_muffin_articles()
    dataset = dataset + get_muffin_video_transcripts()
    print(f"Number of articles: {len(dataset)}")

    dataset = add_steps_to_dataset(dataset, task)

    # dataset = add_info_labels_to_dataset(dataset, task)

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset


def muffins():
    task = "Making Muffins"
    dataset = get_dataset_muffins(task, "raw")
    taxonomy = construct_step_taxonomy(dataset, task, "stupid")
    dataset = extract_ipos_stupid(dataset, taxonomy, task, "stupid")
    ipo_taxonomy = taxonomize_ipos(dataset, taxonomy, task, "stupid")
    dataset = extract_information_per_ipo(dataset, taxonomy, ipo_taxonomy, task, "stupid")
    print("pre-process done!")

    representation = extract_task_representation(dataset, taxonomy, ipo_taxonomy, task, "stupid-" + AGGREGATION_APPROACH)
    tutorial_urls = []
    url_to_content = {}
    url_to_nice_name = {}
    for idx, article in enumerate(dataset):
        url_to_content[article["url"]] = article["content"]
        tutorial_urls.append(article["url"])
        url_to_nice_name[article["url"]] = f"T{idx}"

def main():
    cross_task(3)
    # muffins()

if __name__ == "__main__":
    main()
