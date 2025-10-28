"""
RAG baseline for evaluating information retrieval performance.

- Includes a generic function that (1) can encode the knowledge base (i.e., a corpus of tutorial videos) into a vector database, and (2) encode the query (which is likely the target tutorial + query) into a vector. Uses TF-IDF or Sentence-BERT or OpenAI embeddings in `./helpers/`.

- Includes a function that can perform RAG retrieval on the vector database using the query vector and respond to the query.
"""
import os
import pickle
import numpy as np

from helpers import perform_embedding
from helpers.bert import mccs
from prompts.rag import get_rag_response_full_tutorial, get_rag_response_tutorial_segment

EMBEDDINGS_PATH = "./static/results/rag/"

def encode_dataset(embedding_method, task, dataset):
    """
    Encode the dataset into a vector database.
    """
    embeddings_path = EMBEDDINGS_PATH + f"{task}_{embedding_method}_dataset_embeddings.pkl"
    if os.path.exists(embeddings_path):
        embeddings = pickle.load(open(embeddings_path, "rb"))
    else:
        texts = [(
            "Tutorial: " + item["content"]
            ) for item in dataset]
        embeddings = perform_embedding(embedding_method, texts)
        pickle.dump(embeddings, open(embeddings_path, "wb"))
    return embeddings

def encode_query(embedding_method, tutorial, query, segment):
    """
    Encode the query into a vector.
    """
    if segment is None:
        texts = ["Tutorial: " + tutorial["content"] + "\n" + "Query: " + query]
    else:
        texts = ["Tutorial: " + tutorial["content"] + "\n" + "Segment: " + segment + "\n" + "Query: " + query]
    embeddings = perform_embedding(embedding_method, texts)
    return embeddings

def perform_retrieval(embedding_method, task, dataset, tutorial, query, segment, k):
    """
    Perform RAG retrieval on the vector database using the query vector and respond to the query.
    """
    dataset_embeddings = encode_dataset(embedding_method, task, dataset)
    query_embeddings = encode_query(embedding_method, tutorial, query, segment)
    
    if k is None:
        k = dataset_embeddings.shape[0]
    
    document_idxs, scores = mccs(dataset_embeddings, query_embeddings, top_k=k + 1) ### k + 1 to later filter out the tutorial itself
    document_idxs = document_idxs.flatten()
    scores = scores.flatten()
    
    documents = []
    for idx, score in zip(document_idxs, scores):
        if dataset[idx]["url"] == tutorial["url"]:
            ### skip the tutorial itself
            print(f"retrieval works okay, because it can retrieve itself")
            continue
        documents.append({
            "title": dataset[idx]["title"],
            "content": dataset[idx]["content"],
            "score": score,
        })
    if len(documents) > k:
        documents = documents[:k]
    return documents

def respond_to_query_rag(embedding_method, task, dataset, tutorial, segment, query, k, doc_score_threshold):
    documents = perform_retrieval(embedding_method, task, dataset, tutorial, query, segment, k)

    filtered_documents = []
    if doc_score_threshold is not None:
        for document in documents:
            if document["score"] >= doc_score_threshold:
                filtered_documents.append(document)
    else:
        filtered_documents = documents
    
    if len(filtered_documents) == 0:
        raise ValueError(f"No documents retrieved.")

    if segment is None:
        return get_rag_response_full_tutorial(task, filtered_documents, tutorial, query)
    else:
        return get_rag_response_tutorial_segment(task, filtered_documents, tutorial, segment, query)

generic_call_rag = lambda embedding_method, task, dataset, tutorial, segment, query, info_type, n, k, doc_score_threshold: respond_to_query_rag(embedding_method, task, dataset, tutorial, segment, query, k, doc_score_threshold)