"""
Implementation of two methods based on the CIM:
- Context similarity retrieval: retrieves IUs with most similar contexts to the target tutorial/segment contexts
- Shortest path retrieval: retrieves IUs that have shortest path to the target tutorial/segment contexts
"""

from helpers.dataset import get_dataset
from src.framework_v0 import construct_cim

def context_similarity_retrieval(task, tutorial, query, segment, n):
    ### segment can be None and n can be None
    dataset = get_dataset(task)
    cim = construct_cim(task, "final")

    ### single response: {"content": "...", "source_doc_idx": "..."}
    pass

generic_call_context_similarity = lambda embedding_method, task, dataset, tutorial, query, segment, n, k, doc_score_threshold: context_similarity_retrieval(embedding_method, task, dataset, tutorial, query, segment, n)

def shortest_path_retrieval(task, tutorial, query, segment, n):
    ### segment can be None and n can be None
    dataset = get_dataset(task)
    cim = construct_cim(task, "final")

    ### single response: {"content": "...", "source_doc_idx": "..."}
    pass

generic_call_shortest_path = lambda embedding_method, task, dataset, tutorial, query, segment, n, k, doc_score_threshold: shortest_path_retrieval(embedding_method, task, dataset, tutorial, query, segment, n)
