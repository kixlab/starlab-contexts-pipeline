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