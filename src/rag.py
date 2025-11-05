"""
RAG baseline for evaluating information retrieval performance.

- Includes a generic function that (1) can encode the knowledge base (i.e., a corpus of tutorial videos) into a vector database, and (2) encode the query (which is likely the target tutorial + query) into a vector. Uses TF-IDF or Sentence-BERT or OpenAI embeddings in `./helpers/`.

- Includes a function that can perform RAG retrieval on the vector database using the query vector and respond to the query.
"""
import os
import pickle
import numpy as np

from helpers import perform_embedding
from helpers.nlp import mccs
from prompts.rag import get_rag_response_full_tutorial, get_rag_response_tutorial_segment

EMBEDDINGS_PATH = "./static/results/rag/"

def encode_dataset(embedding_method, task, dataset):
    """
    Encode the dataset into a vector database.
    TODO: perform chunking
    """
    embeddings_path = EMBEDDINGS_PATH + f"{task}_{embedding_method}_dataset_embeddings.pkl"
    if os.path.exists(embeddings_path):
        result = pickle.load(open(embeddings_path, "rb"))
    else:
        texts = [(
            "Tutorial: " + item["content"]
            ) for item in dataset]
        metadata = [{
            "index": i,
            "title": item["title"],
            "url": item["url"],
        } for i,item in enumerate(dataset)]
        embeddings = perform_embedding(embedding_method, texts)
        result = {
            "embeddings": embeddings,
            "metadata": metadata,
        }
        pickle.dump(result, open(embeddings_path, "wb"))
    return result["embeddings"], result["metadata"]

def encode_query(embedding_method, tutorial, query, segment):
    """
    Encode the query into a vector.
    """
    if segment is None:
        texts = ["Tutorial: " + tutorial["content"]]
        # texts = ["Tutorial: " + tutorial["content"] + "\n" + "Query: " + query]
    else:
        texts = ["Tutorial segment: " + segment["content"]]
        # texts = ["Tutorial: " + tutorial["content"] + "\n" + "Segment: " + segment + "\n" + "Query: " + query]
    embeddings = perform_embedding(embedding_method, texts)
    return embeddings

def perform_retrieval(embedding_method, dataset_embeddings, tutorial, query, segment, k, doc_score_threshold):
    """
    Perform RAG retrieval on the vector database using the query vector and respond to the query.
    """
    query_embeddings = encode_query(embedding_method, tutorial, query, segment)
    
    document_idxs, scores = mccs(dataset_embeddings, query_embeddings, top_k=k)
    document_idxs = document_idxs.flatten()
    scores = scores.flatten()
    
    ### return document indices.

    filtered_document_idxs = []
    if doc_score_threshold is not None:
        for document_idx, score in zip(document_idxs, scores):
            if score >= doc_score_threshold:
                filtered_document_idxs.append(document_idx)
    else:
        filtered_document_idxs = document_idxs
    return filtered_document_idxs

def rerank_documents(dataset, document_idxs, metadata, tutorial, query):
    """
    TODO: Rerank the documents based on the query.
    """
    tutorials = []
    for idx in document_idxs:
        dataset_idx = metadata[idx]["index"] ### TODO: there needs to be more complicated weighting here
        if dataset[dataset_idx]["url"] == tutorial["url"]:
            ### skip the tutorial itself
            print(f"retrieval works okay, because it can retrieve itself")
            continue
        tutorials.append({
            "title": dataset[dataset_idx]["title"],
            "content": dataset[dataset_idx]["content"],
        })
    return tutorials

def respond_to_query_rag(task, tutorials, tutorial, segment, query, gen_model):
    if segment is None:
        return get_rag_response_full_tutorial(task, tutorials, tutorial, query, gen_model)
    else:
        return get_rag_response_tutorial_segment(task, tutorials, tutorial, segment, query, gen_model)

def run_rag(task, dataset, tests, embedding_method, gen_model, k, doc_score_threshold):
    doc_embeddings, metadata = encode_dataset(embedding_method, task, dataset)

    responses = []
    for test in tests:
        # info_type = test["info_type"]
        # n = test["n"]
        tutorial = test["tutorial"]
        segment = test["segment"]
        query = test["query"]

        if k is None or k > len(doc_embeddings):
            k = len(doc_embeddings)-1

        document_idxs = perform_retrieval(embedding_method, doc_embeddings, tutorial, query, segment, k + 1, doc_score_threshold)
        tutorials = rerank_documents(dataset, document_idxs, metadata, tutorial, query)
        response = respond_to_query_rag(task, tutorials, tutorial, segment, query, gen_model)
        responses.append(response)
    return responses

generic_call_rag = lambda task, version, dataset, tests, embedding_method, gen_model, k, doc_score_threshold: run_rag(task, dataset, tests, embedding_method, gen_model, k, doc_score_threshold)