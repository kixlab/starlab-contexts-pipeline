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

    ## Calculate cosine similarity between query_embeddings and embeddings
    cos_scores = np.dot(query_embeddings, embeddings.T)
    top_results_per_query = cos_scores.argsort()[:,-1].tolist()
    scores = cos_scores[
        np.arange(len(query_embeddings)),
        top_results_per_query
    ].tolist()
    return top_results_per_query, scores

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
