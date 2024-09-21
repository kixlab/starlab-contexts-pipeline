import os

from pathlib import Path

import json

import numpy as np
import pandas as pd

from helpers.howto100m import process_videos, Video

from helpers.bert import bert_embedding
from server.helpers.sklearn import tfidf_embedding
from server.helpers.sklearn import reduce_dim, standardize
from helpers.clip import clip_embed_image, clip_embed_text

import matplotlib.pyplot as plt

RESULTS = Path("static/figures/")

def format_rgba(color):
    return f"rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, 1)"

def embed_texts(texts, method="bert", truncate=0):
    if truncate > 0:
        for i in range(len(texts)):
            texts[i] = texts[i][:truncate]
    if method == "tfidf":
        return tfidf_embedding(texts)
    if method == "clip":
        return clip_embed_text(texts)
    return bert_embedding(texts)

def embed_images(images, method="clip"):
    if method == "clip":
        return clip_embed_image(images)
    return []

def draw_embeddings(task_id, embeddings, contents, labels, colors, figure_name="embedding"):
    if os.path.exists(RESULTS / f"{task_id}") == False:
        os.makedirs(RESULTS / f"{task_id}")

    embeddings = np.array(embeddings)
    embeddings = standardize(embeddings)
    
    data = pd.DataFrame(embeddings, columns=['x', 'y'])
    data['content'] = [content for content in contents]
    data['label'] = [label for label in labels]
    data['color'] = [color for color in colors]

    fig, ax = plt.subplots()
    data.plot.scatter(x='x', y='y', ax=ax, color=data['color'])
    for i, txt in enumerate(data['label']):
        ax.annotate(txt, (data['x'].iloc[i], data['y'].iloc[i]))


    ### save the plot
    plt.savefig(RESULTS / f"{task_id}" / f"{figure_name}.png")

    formatted_data = []
    for idx, row in data.iterrows():
        formatted_data.append({
            "id": row['label'],
            "x": row['x'],
            "y": row['y'],
            "color": format_rgba(row['color']),
            "data": {
                "longtext": row["content"],
            },
        })

    with open(RESULTS / f"{task_id}" / f"{figure_name}.json", 'w') as f:
        json.dump(formatted_data, f, indent=4)

def get_representation_video(
    task_id="11967",
    n=100,
):
    videos = process_videos(
        n=n,
        task_id=task_id,
        cat1="Food and Entertaining",
        cat2="Recipes",
    )

    transcripts = []
    contents = []
    labels = []
    colors = []

    for video in videos:
        transcript = video.extract_transcript_str()
        transcripts.append(transcript)
        contents.append({
            "content": transcript,
            "label": "all",
            "step_assignment_explanation": "",
            "assignment": {},
        })
        labels.append(video.video_id)
        colors.append(np.random.rand(3,))
    
    return {
        "transcripts": transcripts,
        "contents": contents,
        "labels": labels,
        "colors": colors,
    }

def get_representation_per_step(
    task_id="11967",
    steps_to_consider=None,
):
    assignments_filepath = f"step_data/{task_id}/assignments.json"
    dimensions_filepath = f"step_data/{task_id}/variability_dimensions.json"

    videos = {}
    dimensions = {}

    with open(assignments_filepath, 'r') as f:
        videos = json.load(f)
    
    with open(dimensions_filepath, 'r') as f:
        dimensions = json.load(f)

    transcripts = []
    dims = []
    contents = []
    labels = []
    colors = []
    for video_id, video in videos.items():
        for step_id, step in video.items():
            step_id = step_id.lower()
            if (steps_to_consider is not None
                and step_id not in steps_to_consider):
                print(step_id, steps_to_consider)
                continue
            dims.append("")
            for dimension in step["assignment"].values():
                dims[-1] += f"{dimension['label']}: {dimension['value']}\n"

            transcripts.append(f"{step['content']}")

            contents.append(step)
            labels.append(f"{video_id}-{step_id}")
            colors.append(np.random.rand(3,))

    dim_defs = []
    for dimension in dimensions.values():
        dim_defs.append(f"{dimension['label']}: {dimension['description']}")

    return {
        "transcripts": transcripts,
        "dims": dims,
        "dim_defs": dim_defs,
        "contents": contents,
        "labels": labels,
        "colors": colors,
    }


def generate_embeddings(
    fit_embeddings,
    transform_embeddings,
    method,
    truncate,
):  
    fit_embeddings = embed_texts(
        fit_embeddings,
        method, 
        truncate
    )
    transform_embeddings = embed_texts(
        transform_embeddings,
        method,
        truncate
    )

    embeddings, pca = reduce_dim(
        fit_embeddings,
        transform_embeddings,
        n_components=2
    )

    return embeddings, pca

def augment_embeddings(
    representation,
    augmentation,
):
    embeddings = []
    if augmentation == "dims":
        for i in range(len(representation["transcripts"])):
            transcript = representation["transcripts"][i]
            dims = representation["dims"][i]
            embeddings.append(f"{dims}\n{transcript}")
    else:
        for transcript in representation["transcripts"]:
            embeddings.append(transcript)
    return embeddings
        

def __process(
    task_id="11967", 
    representation=[],
    granularity="video",
    augmentation="", # "dims"
    embedding_method="bert",
    reduction_method="pca-merged", # pca-seperate
    truncate=0,
):
    if len(representation) == 0:
        return
    filename_prefix = f"{granularity}_{task_id}_{embedding_method}_{reduction_method}_{augmentation}_{truncate}"
    print("Filename Prefix: ", filename_prefix)
    
    if reduction_method.endswith("seperate"):
        if "dim_defs" not in representation or embedding_method == "tfidf":
            print("Skipping")
            return

    if reduction_method.endswith("seperate"):
        ### define fit and transform embeddings
        fit_embeddings = representation["dim_defs"]
        transform_embeddings = augment_embeddings(representation, augmentation)
                
        embeddings, pca = generate_embeddings(
            fit_embeddings,
            transform_embeddings,
            embedding_method,
            truncate,
        )
    else:
        embeddings = augment_embeddings(representation, augmentation)

        embeddings, pca = generate_embeddings(
            embeddings,
            embeddings,
            embedding_method,
            truncate,
        )

    print("Explained Variance Ratio: ", pca.explained_variance_ratio_)
    print("Eigenvalues: ", pca.singular_values_)
    print()

    contents = representation["contents"]
    labels = representation["labels"]
    colors = representation["colors"]

    draw_embeddings(task_id, embeddings, contents, labels, colors, filename_prefix)

def process(
    task_id="11967",
    granularity="video",
    n=5,
):
    if granularity == "video":
        representation = get_representation_video(task_id, n=n)
    elif granularity == "step":
        representation = get_representation_per_step(task_id)
    elif granularity.startswith("step"):
        representation = get_representation_per_step(task_id, granularity)
    methods = [
        "bert",
        "tfidf",
        # "clip",
    ]

    reduction_methods = [
        "pca-seperate",
        "pca-merged",
    ]

    augmentations = [
        "",
        "dims",
    ]

    truncates = [
       0, # 50, 100
    ]

    if granularity == "video":
        for truncate in truncates:
            for method in methods:
                __process(
                    task_id=task_id,
                    representation=representation,
                    granularity=granularity,
                    augmentation=augmentations[0],
                    embedding_method=method,
                    reduction_method=reduction_methods[1],
                    truncate=truncate,
                )
    else:
        for truncate in truncates:
            for embedding_method in methods:
                for reduction_method in reduction_methods:
                    for augmentation in augmentations:
                        __process(
                            task_id=task_id,
                            representation=representation,
                            granularity=granularity,
                            augmentation=augmentation,
                            embedding_method=embedding_method,
                            reduction_method=reduction_method,
                            truncate=truncate,
                        )

def main():

    task_ids = [
        "11967",
        "13630",
    ]
    # process(
    #     task_id=task_ids[0],
    #     granularity="step",
    #     n=5,
    # )
    # process(
    #     task_id=task_ids[0],
    #     granularity="video",
    #     n=5,
    # )
    process(
        task_id=task_ids[1],
        granularity="step",
        n=5,
    )
    process(
        task_id=task_ids[1],
        granularity="video",
        n=5,
    )

if __name__ == "__main__":
    main()
