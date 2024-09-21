import os

from pathlib import Path

import json

import numpy as np
import pandas as pd

from helpers.bert import bert_embedding
from helpers.sklearn import tfidf_embedding
from helpers.sklearn import reduce_dim, standardize
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