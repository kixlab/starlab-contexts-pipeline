import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def bert_embedding(texts):
    if len(texts) == 0:
        return []

    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "
    embeddings = model.encode(texts)
    return embeddings

def find_most_similar(embeddings, query_embeddings):
    """
    embeddings: List of embeddings (Tensor)
    query_embeddings: Query embeddings (Tensor)
    """

    ## Calculate cosine similarity between query_embeddings and embeddings
    cos_scores = np.dot(query_embeddings, embeddings.T)
    top_results_per_query = cos_scores.argsort()[:,-1].tolist()
    scores = cos_scores[
        np.arange(len(query_embeddings)),
        top_results_per_query
    ].tolist()
    return top_results_per_query, scores

def clustering_custom(texts, similarity_threshold):
    """
    cluster texts that have `high` similarity
    """

    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    labels = []
    embeddings = bert_embedding(texts)

    similarities = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarities[i][j] = np.dot(embeddings[i], embeddings[j])
            similarities[j][i] = similarities[i][j]
    

    labels = [i for i in range(len(texts))]
    visited = [False for _ in range(len(texts))]
    for i in range(len(texts)):
        if visited[i]:
            continue
        visited[i] = True
        for j in range(i+1, len(texts)):
            if visited[j]:
                continue
            if similarities[i][j] >= similarity_threshold:
                visited[j] = True
                labels[j] = labels[i]
    return labels