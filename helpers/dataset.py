import json
import os

from helpers.video_obj import Video


DATASETS_PATH = "./static/results/datasets/"
MIN_VIDEOS = 20

MUFFIN_TASK = "Making Muffins"

"""
Make French Toast			10 steps / 272 videos
Make Irish Coffee			5 steps / 248 videos
Change a Tire				11 steps / 119 videos
Build (sim.) Floating Shelves		5 steps / 173 videos
"""
CROSS_TASK_TASKS = [
    "Change a Tire",
    "Build Simple Floating Shelves",
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

BIG_CUSTOM_TASKS = [
    "How to Make a Sushi Roll",
    "How to Make Caramel Apples",
    "How to Grill Steak",
    "How to Grow a Pumpkin",
    "How to Clean a Glass Top Stove",
]

SUBGOAL_DESCRIPTION = """
Subgoal: Objective of a subsection.
Example: "Now for the intricate layer that will give me the final webbing look."
"""

INSTRUCTION_DESCRIPTION = """
Instruction: Actions that the instructor performs to complete the task.
Example: "We're going to pour that into our silicone baking cups."
"""

TOOL_DESCRIPTION = """
Tool: Introduction of the materials, ingredients, and equipment to be used.
Example: "I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape."
"""


METHOD_DESCRIPTION = SUBGOAL_DESCRIPTION + INSTRUCTION_DESCRIPTION + TOOL_DESCRIPTION

TIP_DESCRIPTION = """
Tip: Additional instructions or information that makes instructions easier, faster, or more efficient.
Example: "I find that it's easier to do just a couple of layers at a time instead of all four layers at a time."
"""

WARNING_DESCRIPTION = """
Warning: Actions that should be avoided.
Example: "I don't know but I would say avoid using bleach if you can."
"""

SUPPLEMENTARY_DESCRIPTION = TIP_DESCRIPTION + WARNING_DESCRIPTION

JUSTIFICATION_DESCRIPTION = """
Justification: Reasons why the instruction was performed.
Example: "Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly."
"""

EFFECT_DESCRIPTION = """
Effect: Consequences of the instruction.
Example: "And these will overhang a little to help hide the gap."
"""
EXPLANATION_DESCRIPTION = JUSTIFICATION_DESCRIPTION + EFFECT_DESCRIPTION

STATUS_DESCRIPTION = """
Status: Descriptions of the current state of the target object.
Example: "Something sticky and dirty all through the back seat."
"""

CONTEXT_DESCRIPTION = """
Context: Descriptions of the method or the setting.
Example: "[...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch."
"""
TOOL_SPECIFICATION_DESCRIPTION = """
Tool Specification: Descriptions of the tools and equipment.
Example: "These are awesome beans, creamy texture, slightly nutty loaded with flavor."
"""

DESCRIPTION_DESCRIPTION = STATUS_DESCRIPTION + CONTEXT_DESCRIPTION + TOOL_SPECIFICATION_DESCRIPTION

OUTCOME_DESCRIPTION = """
Outcome: Descriptions of the final results of the procedure.
Example: "And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list."
"""

REFLECTION_DESCRIPTION = """
Reflection: Summary, evaluation, and suggestions for the future about the overall procedure.
Example: "However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try."
"""

CONCLUSION_DESCRIPTION = OUTCOME_DESCRIPTION + REFLECTION_DESCRIPTION

IMPORTANT_TYPE_DESCRIPTIONS_COARSE = {
    "Method": METHOD_DESCRIPTION,
    "Supplementary": SUPPLEMENTARY_DESCRIPTION,
    "Explanation": EXPLANATION_DESCRIPTION,
    "Description": DESCRIPTION_DESCRIPTION,
    "Conclusion": CONCLUSION_DESCRIPTION,
}

IMPORTANT_TYPE_DESCRIPTIONS_FINE = {
    "Method - Subgoal": SUBGOAL_DESCRIPTION,
    "Method - Instruction": INSTRUCTION_DESCRIPTION,
    "Method - Tool": TOOL_DESCRIPTION,
    "Supplementary - Tip": TIP_DESCRIPTION,
    "Supplementary - Warning": WARNING_DESCRIPTION,
    "Explanation - Justification": JUSTIFICATION_DESCRIPTION,
    "Explanation - Effect": EFFECT_DESCRIPTION,
    "Description - Status": STATUS_DESCRIPTION,
    "Description - Context": CONTEXT_DESCRIPTION,
    "Description - Tool Specification": TOOL_SPECIFICATION_DESCRIPTION,
    "Conclusion - Outcome": OUTCOME_DESCRIPTION,
    "Conclusion - Reflection": REFLECTION_DESCRIPTION,
}

IMPORTANT_TYPES_COARSE = ["Method", "Supplementary", "Explanation", "Description"]

IMPORTANT_TYPES_FINE = [
    "Method - Subgoal", "Method - Instruction", "Method - Tool",
    "Supplementary - Tip", "Supplementary - Warning",
    "Explanation - Justification", "Explanation - Effect",
    "Description - Status", "Description - Context", "Description - Tool Specification",
    "Conclusion - Outcome", "Conclusion - Reflection",
]

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

def get_muffin_video_transcripts():
    library_metadata = {}
    with open("./static/datasets/custom-dataset/metadata.json") as f:
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

def get_dataset_muffins(task, version=""):
    dataset_filepath = f"{DATASETS_PATH}{task.replace(' ', '_').lower()}_{version}.json"
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
                
                segments = []
                for annotation in video["annotations"]:
                    segments.append({
                        "label": annotation["step"],
                        "start": annotation["start"],
                        "end": annotation["end"],
                    })
                dataset.append({
                    "id": video["video_id"],
                    "url": video["video_url"],
                    "title": task,
                    "original_title": _task["task_name"],
                    "content": content,
                    "transcript": transcript,
                    "segments": segments,
                })

    ### check if content is enough
    filtered_dataset = []
    for article in dataset:
        if len(article["content"]) < 100:
            continue
        filtered_dataset.append(article)
    dataset = filtered_dataset

    return dataset

def preprocess_cross_task(task, version=""):
    dataset_filepath = f"{DATASETS_PATH}{task.replace(' ', '_').lower()}_{version}.json"
    if os.path.exists(dataset_filepath):
        with open(dataset_filepath) as f:
            dataset = json.load(f)
        return dataset


    dataset = get_dataset_cross_task(task)
    print(f"Dataset for {task}: {len(dataset)}")

    with open(dataset_filepath, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset

def preprocess_custom_dataset(task, version=""):
    dataset_filepath = f"{DATASETS_PATH}{task.replace(' ', '_').lower()}_{version}.json"
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

def get_dataset(task):
    if task == MUFFIN_TASK:
        return get_dataset_muffins(task, "framework_raw")
    elif task in CROSS_TASK_TASKS:
        return preprocess_cross_task(task, "framework_raw")
    elif task in CUSTOM_TASKS:
        return preprocess_custom_dataset(task, "framework_raw")

"""
<type>
    <title> Greeting </title>
    <definition> Opening and closing remarks. </definition>
    <subtypes>
        <subtype>
            <title> Opening </title>
            <definition> Starting remarks and instructor/channel introductions. </definition>
            <example> Hey, what's up you guys, Chef [...] here. </example>
        </subtype>
        <subtype>
            <title> Closing </title>
            <definition> Parting remarks and wrap-up. </definition>
            <example> Stay tuned, we'll catch you all later. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Overview </title>
    <definition> Main purpose of the video and its descriptions. </definition>
    <subtypes>
        <subtype>
            <title> Goal </title>
            <definition> Main purpose of the video and its descriptions. </definition>
            <example> Today, I'll show you a special technique which is totally special and about image pressing. </example>
        </subtype>
        <subtype>
            <title> Motivation </title>
            <definition> Reasons or background information on why the video was created. </definition>
            <example> [...] Someone is making a very special valentine's day meal for another certain special someone. </example>
        </subtype>
        <subtype>
            <title> Briefing </title>
            <definition> Rundown of how the goal will be achieved. </definition>
            <example> I'm pretty sure that just taking a pencil and putting it over the front and then putting a bunch of rubber bands around the pencil [...] that's going to do it. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Method </title>
    <definition> Actions that the instructor performs to complete the task. </definition>
    <subtypes>
        <subtype>
            <title> Subgoal </title>
            <definition> Objective of a subsection. </definition>
            <example> Now for the intricate layer that will give me the final webbing look. </example>
        </subtype>
        <subtype>
            <title> Instruction </title>
            <definition> Actions that the instructor performs to complete the task. </definition>
            <example> We're going to pour that into our silicone baking cups. </example>
        </subtype>
        <subtype>
            <title> Tool </title>
            <definition> Introduction of the materials, ingredients, and equipment to be used. </definition>
            <example> I'm also going to use a pair of scissors, a glue stick, some fancy tape or some regular tape. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Supplementary </title>
    <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
    <subtypes>
        <subtype>
            <title> Tip </title>
            <definition> Additional instructions or information that makes instructions easier, faster, or more efficient. </definition>
            <example> I find that it's easier to do just a couple of layers at a time instead of all four layers at a time. </example>
        </subtype>

        <subtype>
            <title> Warning </title>
            <definition> Actions that should be avoided. </definition>
            <example> I don't know but I would say avoid using bleach if you can. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Explanation </title>
    <definition> Reasons why the instruction was performed. </definition>
    <subtypes>
        <subtype>
            <title> Justification </title>
            <definition> Reasons why the instruction was performed. </definition>
            <example> Because every time we wear our contact lenses, makeup and even dirt particles [...] might harm our eyes directly. </example>
        </subtype>
        <subtype>
            <title> Effect </title>
            <definition> Consequences of the instruction. </definition>
            <example> And these will overhang a little to help hide the gap. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Description </title>
    <definition> Descriptions of the current state of the target object. </definition>
    <subtypes>
        <subtype>
            <title> Status </title>
            <definition> Descriptions of the current state of the target object. </definition>
            <example> Something sticky and dirty all through the back seat. </example>
        </subtype>
        <subtype>
            <title> Context </title>
            <definition> Descriptions of the method or the setting. </definition>
            <example> [...] The process of putting on a tip by hand [...] takes a lot of patience but it can be done if you're in a pinch. </example>
        </subtype>
        <subtype>
            <title> Tool Specification </title>
            <definition> Descriptions of the tools and equipment. </definition>
            <example> These are awesome beans, creamy texture, slightly nutty loaded with flavor. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Conclusion </title>
    <definition> Descriptions of the final results of the procedure. </definition>
    <subtypes>
        <subtype> 
            <title> Outcome </title>
            <definition> Descriptions of the final results of the procedure. </definition>
            <example> And now we have a dinosaur taggy blanket that wrinkles, so a fun gift for any baby on your gift giving list. </example>
        </subtype>
        <subtype>
            <title> Reflection </title>
            <definition> Summary, evaluation, and suggestions for the future about the overall procedure. </definition>
            <example> However, I am still concerned about how safe rubbing alcohol actually is to use so maybe next time, I will give vodka a try. </example>
        </subtype>
    </subtypes>
</type>

<type>
    <title> Miscellaneous </title>
    <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
    <subtypes>
        <subtype>
            <title> Side Note </title>
            <definition> Personal stories, jokes, user engagement, and advertisements. </definition>
            <example> Tristan is back from basketball - He made it on the team so it's pretty exciting. </example>
        </subtype>
        <subtype>
            <title> Self-promotion </title>
            <definition> Promotion of the instructor of the channel (i.e. likes, subscription, notification, or donations). </definition>
            <example> So if you like this video, please give it a thumbs up and remember to subscribe. </example>
        </subtype>
        <subtype>
            <title> Bridge </title>
            <definition> Meaningless phrases or expressions that connect different sections. </definition>
            <example> And we're going to go ahead and get started. </example>
        </subtype>
        <subtype>
            <title> Filler </title>
            <definition> Conventional filler words. </definition>
            <example> Whoops. </example>
        </subtype>
    </subtypes>
</type>
"""