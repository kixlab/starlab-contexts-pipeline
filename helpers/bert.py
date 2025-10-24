import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from stop_words import get_stop_words


### fine-tune the model
model = SentenceTransformer("all-MiniLM-L6-v2")
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=1,
#     warmup_steps=100,
#     optimizer_params={'lr': 1e-4},
# )

en_stop_words = get_stop_words('en')

def bert_embedding(texts):
    if len(texts) == 0:
        return np.array([])

    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "
    embeddings = model.encode(texts)
    return embeddings

def tfidf_embedding(texts):
    if len(texts) == 0:
        return np.array([])
    vectorizer = TfidfVectorizer(
        stop_words=en_stop_words,
        max_features=100,
        max_df=0.9,
        # min_df=0.2,
        smooth_idf=True,
        norm='l2',
        ngram_range=(1, 2),
    )
    embeddings = vectorizer.fit_transform(texts)
    print("FEATURE_NAMES:")
    print(vectorizer.get_feature_names_out())
    return np.array(embeddings.toarray())

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
    if top_k is None:
        top_k = embeddings.shape[0]
    else:
        top_k = min(top_k, embeddings.shape[0])
    if top_k == 0:
        return [], []
    mccs_scores = np.dot(query_embeddings, embeddings.T)
    top_k_indices_per_query = mccs_scores.argsort()[:,::-1][:,:top_k].tolist()
    scores = np.take_along_axis(mccs_scores, top_k_indices_per_query, axis=-1)
    return top_k_indices_per_query, scores

def clustering_custom(texts, similarity_threshold, embedding_method="bert"):
    """
    cluster texts that have `high` similarity
    """

    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    labels = []
    if embedding_method == "tfidf":
        embeddings = tfidf_embedding(texts)
    else:
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

def hierarchical_clustering(
        texts, embedding_method="bert", linkage="average", n_clusters=None, distance_threshold=None
):
    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    if embedding_method == "tfidf":
        embeddings = tfidf_embedding(texts)
    else:
        embeddings = bert_embedding(texts)

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

def find_most_distant_pair(texts, embedding_method="bert"):
    if len(texts) <= 1:
        return None
    
    if embedding_method == "tfidf":
        embeddings = tfidf_embedding(texts)
    else:
        embeddings = bert_embedding(texts)
    similarities = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarities, -float("inf"))
    max_distance_pair = np.unravel_index(np.argmax(similarities), similarities.shape)
    idx_1 = max_distance_pair[0]
    idx_2 = max_distance_pair[1]
    return idx_1, idx_2