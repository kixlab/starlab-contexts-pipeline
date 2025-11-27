"""
RAG baseline for evaluating information retrieval performance.

- RAG retrieval using TF-IDF embeddings
- TODO: Reranking with BERT (https://github.com/nyu-dl/dl4marco-bert)
"""
import os
import pickle

from helpers import perform_embedding
from helpers.nlp import mccs
from prompts.rag import get_rag_response_request, get_rag_response_response
from prompts.framework_batch import batch_run_lm_calls

EMBEDDINGS_PATH = "./static/results/rag/"

def encode_dataset(task, dataset, embedding_method):
    """
    Encode the dataset into a vector database.
    chunks similar to query size
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
        } for i, item in enumerate(dataset)]
        embeddings = perform_embedding(embedding_method, texts)
        result = {
            "embeddings": embeddings,
            "metadata": metadata,
        }
        pickle.dump(result, open(embeddings_path, "wb"))
    return result["embeddings"], result["metadata"]

# def encode_query(embedding_method, tutorial, query, segment):
#     """
#     Encode the query into a vector.
#     """
#     if segment is None:
#         texts = ["Tutorial: " + tutorial["content"]]
#         # texts = ["Tutorial: " + tutorial["content"] + "\n" + "Query: " + query]
#     else:
#         texts = ["Tutorial segment: " + segment["content"]]
#         # texts = ["Tutorial: " + tutorial["content"] + "\n" + "Segment: " + segment + "\n" + "Query: " + query]
#     embeddings = perform_embedding(embedding_method, texts)
#     return embeddings

def perform_retrieval(dataset_embeddings, metadata, query_tutorial, doc_k, doc_score_threshold):
    """
    ### Find most similar based on tfidf
    """
    query_tutorial_embeddings = []
    for item in metadata:
        dataset_idx = item["index"]
        if item["url"] == query_tutorial["url"]:
            query_tutorial_embeddings.append(dataset_embeddings[dataset_idx])
            break
    if len(query_tutorial_embeddings) == 0:
        raise ValueError(f"Query tutorial not found in dataset: {query_tutorial['url']}")
    
    document_idxs, scores = mccs(dataset_embeddings, query_tutorial_embeddings, top_k=doc_k)
    document_idxs = document_idxs.flatten()
    scores = scores.flatten()
    
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
    TODO: Rerank with BERT (https://github.com/nyu-dl/dl4marco-bert)
    """
    tutorials = []
    for idx in document_idxs:
        dataset_idx = metadata[idx]["index"] ### TODO: there needs to be more complicated weighting here
        if dataset[dataset_idx]["url"] == tutorial["url"]:
            ### skip the tutorial itself
            print(f"retrieval works okay, because it can retrieve itself")
            continue
        tutorials.append({
            "url": dataset[dataset_idx]["url"],
            "title": dataset[dataset_idx]["title"],
            "pieces": dataset[dataset_idx]["pieces"],
        })
    return tutorials

def run_rag(task, dataset, tests, config):
    gen_model = config["gen_model"]
    doc_k = config["doc_k"]
    doc_score_threshold = config["doc_score_threshold"]
    embedding_method = config["embedding_method"]
    doc_embeddings, metadata = encode_dataset(task, dataset, embedding_method)

    if doc_k is None or doc_k > len(doc_embeddings):
        doc_k = len(doc_embeddings)-1

    request_args = []
    for test in tests:
        # info_type = test["info_type"]
        # n = test["n"]
        tutorial = test["tutorial"]
        segment = test["segment"]
        query = test["query"]

        document_idxs = perform_retrieval(doc_embeddings, metadata, tutorial, doc_k + 1, doc_score_threshold)
        tutorials = rerank_documents(dataset, document_idxs, metadata, tutorial, query)
        tutorials = tutorials[:doc_k]
        if len(tutorials) < doc_k:
            print(f"WARNING: Not enough tutorials retrieved for {tutorial['url']}")

        request_args.append({
            "task": task,
            "tutorials": tutorials,
            "tutorial": tutorial,
            "segment": segment,
            "query": query,
            "gen_model": gen_model,
        })
    
    batch_results = batch_run_lm_calls(request_args, get_rag_response_request, get_rag_response_response)

    responses = []
    for result in batch_results:
        responses.append(result)
    return responses

generic_call_rag = lambda task, dataset, tests, config: run_rag(task, dataset, tests, config)