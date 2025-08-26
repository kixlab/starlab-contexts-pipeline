import json
import os
import random
import gzip
import numpy as np
import requests

import webvtt
import textwrap

from bs4 import BeautifulSoup

from helpers.bert import bert_embedding, tfidf_embedding

from helpers import str_to_float

PATH = "./static/results/"

WIKIHOW_CORPUS_PATH = "./static/datasets/wikihow-corpus/"

CUSTOM_DATASET_PATH = "./static/datasets/custom-dataset/"

YT_TEMPORAL_DATASET_PATH = "./static/datasets/yt-temporal-180m/"

SIMILARITY_THRESHOLD = 0.75
MIN_NUMBER_OF_VIDEOS_PER_TASK = 20

random.seed(42)

def get_full_wikihow_corpus():
    data_json = []
    for filename in os.listdir(WIKIHOW_CORPUS_PATH):
        if filename.endswith(".json"):
            if filename.startswith("wikihow"):
                continue
            with open(WIKIHOW_CORPUS_PATH + filename) as f:
                data = json.load(f)
                data_json.append(data)
    return data_json

def crawl_articles(links):
    """
    crawl article content from links with bs4
    """
    articles = {}
    for link in links:
        if link.startswith("http"):
            clean_link = link.split('?')[0]
            if clean_link not in articles:
                articles[clean_link] = []
    for link in articles.keys():
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles[link] = soup.get_text()
        except Exception as e:
            print(f"Error crawling {link}: {e}")
            articles[link] = ""
    return articles

def crawl_youtube_videos(task):        
    pass

def crawl_youtube_comments(video_id):
    pass

def crawl_articles_comments(article_url):
    pass


def process_videos(task, source=[]):
    pass

def process_articles(task):
    pass

def load_yt_temporal_dataset(skip=0, until=10000000):

    def process_webvtt(content):
        """
        content is a string of webvtt format
        """

        vtt = webvtt.from_string(textwrap.dedent(content).strip())

        transcript = []
        prev_line = None

        for caption in vtt:
            passed_prev_line = False
            new_lines = []

            lines = caption.text.split("\n")

            for line in lines:
                if prev_line == line:
                    passed_prev_line = True
                    continue
                if passed_prev_line:
                    new_lines.append(line)
            if passed_prev_line is False:
                new_lines = lines

            if len(new_lines) > 0:
                transcript.append({
                    "start": str_to_float(caption.start),
                    "end": str_to_float(caption.end),
                    "text": "\n".join(new_lines),
                })
                prev_line = new_lines[-1]
        
        return transcript

    videos = []
    for idx, video_path in enumerate(os.listdir(YT_TEMPORAL_DATASET_PATH)):
        if idx < skip:
            continue
        if idx >= until:
            break
        ## remove after debugging
        # if len(videos) > 100:
        #     break
        if video_path.endswith(".json.gz"):
            with gzip.open(YT_TEMPORAL_DATASET_PATH + video_path, "rt") as f:
                video_data = json.load(f)
                if "transcripts" not in video_data:
                    continue
                if "en" not in video_data["transcripts"]:
                    continue
                video_data["transcript"] = process_webvtt(video_data["transcripts"]["en"])
                videos.append({
                    "id": video_data["id"],
                    "title": video_data["title"],
                    "description": video_data["description"],
                    "transcript": video_data["transcript"],
                })
    return videos

def get_videos_per_task(tasks, videos, _embedding=tfidf_embedding, threshold=SIMILARITY_THRESHOLD):

    video_features = [video["title"] for video in videos]
    task_features = [task["title"] for task in tasks]
    all_embeddings = _embedding(video_features + task_features)
    video_embeddings = all_embeddings[:len(videos)]
    task_embeddings = all_embeddings[len(videos):]

    similarities = np.dot(video_embeddings, task_embeddings.T)

    videos_per_task = {}
    for task_idx, _ in enumerate(task_embeddings):
        task_name = tasks[task_idx]["title"]
        videos_per_task[task_name] = []
        task_similarities = similarities[:, task_idx]
        top_k_indices = np.argsort(task_similarities)[::-1]
        if threshold is not None:
            top_k_indices = top_k_indices[task_similarities[top_k_indices] >= threshold]
        for video_idx in top_k_indices:
            similarity = float(task_similarities[video_idx])
            videos_per_task[task_name].append({
                **videos[video_idx],
                "similarity": similarity
            })
    return videos_per_task

def get_videos_per_tasks(all_tasks):
    videos_per_task_path = CUSTOM_DATASET_PATH + f"videos_per_task.json"
    if os.path.exists(videos_per_task_path):
        with open(videos_per_task_path, "r") as f:
            data = json.load(f)
            return data
    
    print("crawling videos per task")

    videos = load_yt_temporal_dataset()
    videos_per_task = get_videos_per_task(all_tasks, videos, _embedding=bert_embedding, threshold=SIMILARITY_THRESHOLD)

    videos_per_task_path = CUSTOM_DATASET_PATH + f"videos_per_task.json"
    with open(videos_per_task_path, "w") as f:
        json.dump(videos_per_task, f, indent=4)
    return videos_per_task

ARTICLES_KEY = "__articles"

def sample_n_tasks(sample_size=100):
    sampled_tasks_path = CUSTOM_DATASET_PATH + f"tasks_per_category_{sample_size}.json"
    if os.path.exists(sampled_tasks_path):
        with open(sampled_tasks_path, "r") as f:
            sampled_tasks_per_category = json.load(f)
            return sampled_tasks_per_category

    ### pick tasks from wikiHow
    wikihow_corpus = get_full_wikihow_corpus()
    categories_tree = {}
    for article in wikihow_corpus:
        categories = article["category_hierarchy"]
        time_updated = article["time_updated"]
        n_views = article["n_views"]
        rating = article["rating"]
        n_votes = rating["n_votes"]
        helpful_percent = rating["helpful_percent"]
        title = article["title"]
        description = article["title_description"]
        url = article["url"]
        video_url = article["video"]
        refs = article["refs"]
        # methods = article["methods"]
        # parts = article["parts"]

        cur_node = categories_tree
        for category in categories:
            if category not in cur_node:
                cur_node[category] = {}
            cur_node = cur_node[category]
        if ARTICLES_KEY not in cur_node:
            cur_node[ARTICLES_KEY] = []
        cur_node[ARTICLES_KEY].append({
            "title": title,
            "time_updated": time_updated,
            "n_views": n_views,
            "n_votes": n_votes,
            "helpful_percent": helpful_percent,
            'url': url,
            'video_url': video_url,
            'description': description,
            'refs': refs,
        })

    def get_articles_for_category_hierarchy(cur_node, depth, prefix_categories, target_categories=None, target_depth=None, task_filters=[]):
        results = []
        if ARTICLES_KEY in cur_node:
            extend = False
            if target_categories is not None and depth >= len(target_categories):
                extend = True
            elif target_depth is not None and depth >= target_depth:
                extend = True
            if extend:
                for article in cur_node[ARTICLES_KEY]:
                    skip = False
                    for filter in task_filters:
                        if filter(article) is False:
                            skip = True
                    if skip is False:
                        results.append((prefix_categories, article))
        
        for category, child in cur_node.items():
            if category == ARTICLES_KEY:
                continue
            if target_categories is not None:
                if depth < len(target_categories) and category != target_categories[depth]:
                    continue
            results.extend(get_articles_for_category_hierarchy(child, depth + 1, prefix_categories + [category], target_categories, target_depth, task_filters))
        return results
    selected_categories = [
        ['Food and Entertaining'],
        ['Home and Garden'],
        ['Hobbies and Crafts'],
        ['Cars & Other Vehicles'],
        ['Pets and Animals'],
        ['Holidays and Traditions'],
        ['Personal Care and Style'],
        ['Sports and Fitness'],
        ['Health'],
        ['Education and Communications'],
        ['Arts and Entertainment'],
        ['Computers and Electronics']        
    ]

    def custom_task_filter(article):
        if article['n_views'] < 100000:
            return False
        if article["video_url"] is None:
            return False
        # if article["video_url"] is not None:
        #     return False
        ### refs from at least 5 different websites
        refs_threshold = 5
        ref_count = {}
        for ref in article["refs"]:
            clean_ref = ref
            if ref.startswith("http"):
                clean_ref = ref.split('?')[0]
            if clean_ref not in ref_count:
                ref_count[clean_ref] = 0
            ref_count[clean_ref] += 1
        if len(ref_count) < refs_threshold:
            return False   
        return True

    sampled_tasks_per_category = {}

    for selected_category_hierarchy in selected_categories:
        tasks = get_articles_for_category_hierarchy(categories_tree, 0, [], target_categories=selected_category_hierarchy, task_filters=[custom_task_filter])
        ### randomly sample 100 tasks
        print(selected_category_hierarchy, len(tasks))
        sampled_idxs = random.sample(range(len(tasks)), min(sample_size, len(tasks)))
        sampled_tasks = []
        for idx in sampled_idxs:
            sampled_tasks.append({
                "category_hierarchy": tasks[idx][0],
                "title": tasks[idx][1]["title"],
                "description": tasks[idx][1]["description"],
                "url": tasks[idx][1]["url"],
                "video_url": tasks[idx][1]["video_url"],
                "time_updated": tasks[idx][1]["time_updated"],
                "n_views": tasks[idx][1]["n_views"],
                "n_votes": tasks[idx][1]["n_votes"],
                "helpful_percent": tasks[idx][1]["helpful_percent"],
                "refs": tasks[idx][1]["refs"],
                "is_goal_driven": 0,
            })

        cur_key = "->".join(selected_category_hierarchy)
        sampled_tasks_per_category[cur_key] = sampled_tasks
    ### save
    with open(sampled_tasks_path, "w") as f:
        json.dump(sampled_tasks_per_category, f, indent=4)
    return sampled_tasks_per_category

def pick_tasks_per_category(sampled_tasks, sample_size=2):
    """
    pick `sample_size` tasks per category
    """
    tasks_per_category_path = CUSTOM_DATASET_PATH + f"tasks_per_category_{sample_size}.json"
    if os.path.exists(tasks_per_category_path):
        with open(tasks_per_category_path, "r") as f:
            tasks_per_category = json.load(f)
            return tasks_per_category

    tasks_per_category = {}
    for category, tasks in sampled_tasks.items():
        suitable_tasks = []
        for task in tasks:
            if task["is_goal_driven"] == 1:
                suitable_tasks.append(task)
        if len(suitable_tasks) > 0:
            sampled_idxs = random.sample(range(len(suitable_tasks)), min(sample_size, len(suitable_tasks)))
            tasks_per_category[category] = []
            for idx in sampled_idxs:
                tasks_per_category[category].append(suitable_tasks[idx])
    
    ### save
    with open(tasks_per_category_path, "w") as f:
        json.dump(tasks_per_category, f, indent=4)

    return tasks_per_category

def get_dataset():
    
    dataset_path = CUSTOM_DATASET_PATH + f"videos_tasks_per_category.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            return json.load(f)

    all_tasks = []
    sampled_tasks = sample_n_tasks(sample_size=10)
    for category, tasks in sampled_tasks.items():
        for task in tasks:
            if task["is_goal_driven"] == 1:
                # articles = crawl_articles(task["refs"])
                all_tasks.append(task)
    
    videos_per_task = get_videos_per_tasks(all_tasks)

    videos_tasks_per_category = {}

    for task_name, videos in videos_per_task.items():
        task_details = None
        for task in all_tasks:
            if task["title"] == task_name:
                task_details = task
        if task_details is None:
            continue
        category = task_details["category_hierarchy"][0]
        if category not in videos_tasks_per_category:
            videos_tasks_per_category[category] = []
        applicable_videos = []
        for video in videos:
            similarity = video["similarity"]
            if similarity < 0.7:
                break
            applicable_videos.append(video)
            
        print(f"total videos: {len(applicable_videos)}")
        videos_tasks_per_category[category].append({
            "task_details": task_details,
            "videos": applicable_videos,
        })

    with open(dataset_path, "w") as f:
        json.dump(videos_tasks_per_category, f, indent=4)

    return videos_tasks_per_category

def main():
    dataset = get_dataset()
    ### print some statistics (2 tasks per category with the most videos)
    for category, tasks_videos in dataset.items():
        ### sort tasks by the number of videos
        tasks_videos.sort(key=lambda x: len(x["videos"]), reverse=True)
        print(category)
        for cur_task_videos in tasks_videos:
            if len(cur_task_videos["videos"]) < MIN_NUMBER_OF_VIDEOS_PER_TASK:
                continue
            print(cur_task_videos["task_details"]["title"])
            print(f"total videos: {len(cur_task_videos['videos'])}")
        print()
if __name__ == "__main__":
    main()