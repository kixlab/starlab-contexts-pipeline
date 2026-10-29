import numpy as np
from sklearn.cluster import AgglomerativeClustering
from helpers import perform_embedding

def find_most_similar(embeddings, query_embeddings):
    """
    embeddings: List of embeddings (Tensor)
    query_embeddings: Query embeddings (Tensor)
    """
    top_k_indices_per_query, scores = mccs(embeddings, query_embeddings, top_k=1)
    return top_k_indices_per_query.flatten(), scores.flatten()

def mccs(embeddings, query_embeddings, top_k):
    """
    Maximum Cosine Similarity Search for each query
    Return N x K matrix (N: number of queries, K: top_k) where each cell is the index of the j-th similar embedding.
    """
    
    top_k = min(top_k, embeddings.shape[0])
    if top_k == 0:
        return [], []

    mccs_scores = np.dot(query_embeddings, embeddings.T)
    top_k_indices_per_query = mccs_scores.argsort()[:,::-1][:,:top_k]
    scores = np.array([row[p] for row, p in zip(mccs_scores, top_k_indices_per_query)])
    return top_k_indices_per_query, scores

def clustering_custom(texts, similarity_threshold, embedding_method="bert"):
    """
    cluster texts that have `high` similarity
    """

    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    labels = []
    embeddings = perform_embedding(embedding_method, texts)

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

def hierarchical_clustering(
        texts, embedding_method="bert", linkage="average", n_clusters=None, distance_threshold=None
):
    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    embeddings = perform_embedding(embedding_method, texts)

    similarities = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarities[i][j] = np.dot(embeddings[i], embeddings[j])
            similarities[j][i] = similarities[i][j]

    # clustering = AgglomerativeClustering(
    #     distance_threshold=similarity_thresh,
    #     n_clusters=None,
    #     linkage=linkage,
    #     metric="precomputed"
    # ).fit(similarities)
    # clusters = clustering.labels_

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric="cosine"
    ).fit(embeddings)
    clusters = clustering.labels_

    ### print minimum distance in a cluster
    embeddings_per_cluster = {}
    for i, cluster in enumerate(clusters):
        if cluster not in embeddings_per_cluster:
            embeddings_per_cluster[cluster] = []
        embeddings_per_cluster[cluster].append(i)
    
    for cluster in embeddings_per_cluster:
        print(f"Cluster {cluster}:")
        embedding_ids = embeddings_per_cluster[cluster]
        min_dist = 1e9
        if len(embedding_ids) == 1:
            print("\tSingle element")
            continue
        for i in range(len(embedding_ids)):
            for j in range(i+1, len(embedding_ids)):
                min_dist = min(min_dist, similarities[embedding_ids[i]][embedding_ids[j]])
        print(f"\tMin dist: {min_dist}")

    return clusters.tolist()

def find_most_distant_items(texts, count=2, embedding_method="bert", additional_labels=None):
    """
    Algorithm that finds the most semantically distant items.
    Returns a list of indices of the most distant items.

    Specifically, we want maximize the minimum pairwise distance between the items (i.e., minimize the maximum pairwise similarity).

    If additional_labels are provided, the selected items should have different labels.
    """
    if len(texts) < 2:
        return [i for i in range(len(texts))]
    
    embeddings = perform_embedding(embedding_method, texts)
    
    similarities = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarities, float("inf"))
    
    # Greedy algorithm to maximize total pairwise distance (minimize pairwise similarities)

    # Start with the pair that has the highest distance (lowest similarity)
    has_label = {}
    best_indices = []
    similarities_copy = similarities.copy()
    # similarities is cosine similarity, so we want minimal values
    first_idx, second_idx = np.unravel_index(np.argmin(similarities_copy), similarities_copy.shape)
    if additional_labels is not None:
        if additional_labels[first_idx] != additional_labels[second_idx]:
            best_indices = [first_idx, second_idx]
        else:
            print("WARNING: First and second indices have the same label", texts[first_idx], texts[second_idx])
            best_indices = [first_idx]
    else:
        best_indices = [first_idx, second_idx]

    for index in best_indices:
        if additional_labels is not None:
            has_label[additional_labels[index]] = True

    while len(best_indices) < count:
        # For each item not in best_indices, calculate the minimum similarity to items already selected
        candidates = [i for i in range(len(texts)) if i not in best_indices and (additional_labels is None or additional_labels[i] not in has_label.keys())]
        best_candidate = None
        minimum_max_similarity = float("inf")
        for c in candidates:
            max_similarity = np.max([similarities[c][i] for i in best_indices])
            if max_similarity < minimum_max_similarity:
                minimum_max_similarity = max_similarity
                best_candidate = c
        if best_candidate is None:
            break
        best_indices.append(best_candidate)
        if additional_labels is not None:
            has_label[additional_labels[best_candidate]] = True
    return best_indices