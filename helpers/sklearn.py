import numpy as np
import nltk

from sklearn.decomposition import PCA

def standardize(embeddings):
    std = embeddings.std(axis=0)
    mean = embeddings.mean(axis=0)
    if (std == 0).any():
        return embeddings
    return (embeddings - mean) / std

def reduce_dim(fit_embeddings, transform_embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(fit_embeddings)
    reduced_embeddings = pca.transform(transform_embeddings)

    return reduced_embeddings, pca

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import zeros
from nltk.corpus import stopwords

nltk.download('stopwords')

def tfidf_embedding(texts): 
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        ngram_range=(1, 2), max_features=1000
    )
    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "

    X = vectorizer.fit_transform(texts)
    embeddings = X.toarray()
    return embeddings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull

from helpers.bert import bert_embedding


def k_means_clustering(
        embeddings,
        n_clusters=3,
):
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=10).fit(embeddings)

    s_score = silhouette_score(embeddings, kmeans.labels_, metric='euclidean')
    
    return kmeans.labels_, kmeans.inertia_, s_score

def cluster_texts(texts):
    """
    Returns labels, inertia & silhouette score for each n_clusters
    """
    COEFFICIENT = 2 ### coefficient for elbow point: a point where adding new clusters does not improve the silhouette score by a factor of COEFFICIENT

    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    labels = []
    inertias = []
    s_scores = []
    embeddings = bert_embedding(texts)

    for n_clusters in range(2, len(texts)-1):
        labels_, inertia, s_score = k_means_clustering(embeddings, n_clusters=n_clusters)
        labels.append(labels_)
        inertias.append(inertia)
        s_scores.append(s_score)

    
    points = np.array([[i, s_scores[i]] for i in range(len(s_scores))])

    # Compute the convex hull
    hull = ConvexHull(points)
    
    vertices = list(hull.vertices)
    upper_hull = []
    r = np.argmin(points[vertices, 0])
    l = np.argmax(points[vertices, 0])

    if l < r:
        for i in range(l, r + 1):
            upper_hull.append(vertices[i])
    else:
        for i in range(l, len(vertices)):
            upper_hull.append(vertices[i])
        for i in range(r + 1):
            upper_hull.append(vertices[i])

    hull_points = points[upper_hull]

    hull_points = hull_points[np.argsort(hull_points[:, 0])]

    max_y = np.max(hull_points[:, 0])
    x = hull_points[:, 0] / max_y
    y = hull_points[:, 1]

    # Compute the first derivative (dy/dx)
    dy = np.gradient(y, x)

    # Compute the elbow point
    elbow_point = np.argmax(y)

    while elbow_point > 0 and dy[elbow_point] * COEFFICIENT < dy[elbow_point - 1]:
        elbow_point -= 1

    if elbow_point < 0:
        print("WARNING: Elbow point not found, using the first hull vertice")
        elbow_point = 0
    best_n_clusters = int(hull_points[elbow_point][0]) + 2
    
    ### delete later
    print("# Clusters: ", best_n_clusters)
    print("Inertias: ", inertias)
    print("S Scores: ", s_scores)

    return labels[best_n_clusters-2]

def cluster_tagged_texts(texts, tags, tries=5):
    # print("## Tags: ", tags)
    """
    the smallest number of clusters such that each cluster does not contain same tagged texts
    """

    if len(texts) <= 1:
        return [0 for _ in range(len(texts))]
    
    labels = []
    best_n_clusters = 2
    # inertias = []
    # s_scores = []
    embeddings = bert_embedding(texts)

    left = 2
    right = len(texts)-1
    while left <= right:
        mid = (left + right) // 2
        
        tried = 0
        while tried < tries:
            tried += 1
            labels_, _, _ = k_means_clustering(embeddings, n_clusters=mid)
            
            proper_clusters = True
            tags_set = {}
            for index, tag in enumerate(tags):
                if tag not in tags_set:
                    tags_set[tag] = set()
                if labels_[index] in tags_set[tag]:
                    print("## Violation 1: ", tag, texts[index])
                    for i in range(index):
                        if labels_[i] == labels_[index] and tags[i] == tag:
                            print("## Violation 2: ", tag, texts[i])
                            break
                    print()
                    proper_clusters = False
                    break
                tags_set[tag].add(labels_[index])
            if proper_clusters:
                labels = labels_
                best_n_clusters = mid
                break
        if best_n_clusters == mid:
            right = mid - 1
        else:
            left = mid + 1

    # for n_clusters in range(2, len(texts)-1):
    #     labels_, inertia, s_score = k_means_clustering(embeddings, n_clusters=n_clusters)
    #     inertias.append(inertia)
    #     s_scores.append(s_score)

    #     proper_clusters = True
    #     tags_set = {}
    #     for index, tag in enumerate(tags):
    #         if tag not in tags_set:
    #             tags_set[tag] = set()
    #         if labels_[index] in tags_set[tag]:
    #             print("## Violation 1: ", tag, texts[index])
    #             for i in range(index):
    #                 if labels_[i] == labels_[index] and tags[i] == tag:
    #                     print("## Violation 2: ", tag, texts[i])
    #             proper_clusters = False
    #             break
    #         tags_set[tag].add(labels_[index])
    #     if proper_clusters:
    #         labels = labels_
    #         best_n_clusters = n_clusters
    #         break
    
    if len(labels) == 0:
        labels = [i for i in range(len(texts))]
        best_n_clusters = len(texts)
        # inertias.append(0)
        # s_scores.append(0)

    ### delete later
    print("# Clusters: ", best_n_clusters)
    # print("Inertias: ", inertias)
    # print("S Scores: ", s_scores)

    return labels