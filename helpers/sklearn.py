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

def extract_keysteps(step_sequences, min_clique_size=1):
    """
    Extract keysteps similar to https://aclanthology.org/2023.findings-acl.210.pdf
    """
    MAX_CLIQUE_THRESHOLD = 0.9

    all_steps = []
    for sequence in step_sequences:
        all_steps.extend(sequence)
    
    all_steps = list(set(all_steps))
    if len(all_steps) == 0:
        return []
    
    print(all_steps)
    
    embeddings = bert_embedding(all_steps)

    ## identify max_cliques
    max_cliques = []
    for i in range(len(all_steps)):
        for j in range(i+1, len(all_steps)):
            if np.dot(embeddings[i], embeddings[j]) > MAX_CLIQUE_THRESHOLD:
                max_cliques.append((i, j))

    ## identify keysteps (cluster of steps within a max_clique)
    keysteps = []
    visited = [False for _ in range(len(all_steps))]
    for i, j in max_cliques:
        if not visited[i]:
            cluster = [i]
            visited[i] = True
            for k, l in max_cliques:
                if k == i and not visited[l]:
                    cluster.append(l)
                    visited[l] = True
            keysteps.append(cluster)
    
    ## filter out keysteps with less than min_clique_size
    keysteps = [keystep for keystep in keysteps if len(keystep) >= min_clique_size]

    ## calculate sequence_overlap for each keystep pair (i.e., the number of step_sequences that contain both keysteps)
    sequence_overlap = np.zeros((len(keysteps), len(keysteps)))
    for i in range(len(keysteps)):
        for j in range(i+1, len(keysteps)):
            cooccurence = 0
            for sequence in step_sequences:
                occur = False
                for k in keysteps[i]:
                    if all_steps[k] in sequence:
                        for l in keysteps[j]:
                            if all_steps[l] in sequence:
                                occur = True
                                break
                    if occur:
                        break
                if occur:
                    cooccurence += 1

            sequence_overlap[i][j] = cooccurence
    
    ## merge keysteps with low sequence_overlap but high similarity

    merged_keysteps = []
    visited = [False for _ in range(len(keysteps))]
    for i in range(len(keysteps)):
        if visited[i]:
            continue
        max_similarity_with_different_keystep = 0
        for j in range(0, len(keysteps)):
            if i == j:
                continue
            if sequence_overlap[i][j] == 0:
                continue
            cur_max_similarity = 0
            for k in keysteps[i]:
                for l in keysteps[j]:
                    cur_max_similarity = max(cur_max_similarity, np.dot(embeddings[k], embeddings[l]))
            max_similarity_with_different_keystep = max(max_similarity_with_different_keystep, cur_max_similarity)


        cluster = [i]
        visited[i] = True
        for j in range(i+1, len(keysteps)):
            if visited[j]:
                continue
            ## merge if sequence_overlap is zero but similarity is relatively high
            if sequence_overlap[i][j] > 0:
                continue
            cur_max_similarity = 0
            for k in keysteps[i]:
                for l in keysteps[j]:
                    cur_max_similarity = max(cur_max_similarity, np.dot(embeddings[k], embeddings[l]))
            if cur_max_similarity > max_similarity_with_different_keystep:
                cluster.append(j)
                visited[j] = True
        new_keysteps = []
        for idx in cluster:
            new_keysteps.extend(keysteps[idx])
        merged_keysteps.append(new_keysteps)
    
    ### print merged keysteps for each label
    for i, keysteps in enumerate(merged_keysteps):
        print("## Keystep ", i, ": ")
        for k in keysteps:
            print(f"- {all_steps[k]}")
    
    relabled_step_sequences = []
    for sequence in step_sequences:
        new_sequence = []
        for step in sequence:
            found = False
            for i, keysteps in enumerate(merged_keysteps):
                for k in keysteps:
                    if step == all_steps[k]:
                        new_sequence.append(i)
                        found = True
                        break
                if found:
                    break
            if not found:
                print("## Error: ", step)
        relabled_step_sequences.append(new_sequence)
    
    return relabled_step_sequences


