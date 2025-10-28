import json

from src.framework import construct_cim
from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS

from src.cim_methods import context_similarity_retrieval

from helpers.dataset import get_dataset

def run_framework(task):
    dummy = "segmentation_v1"
    dataset = get_dataset(task)
    _ = construct_cim(task, dataset, dummy)
    print("COMPLETED!")

def run_context_similarity(task):
    dataset = get_dataset(task)
    
    embedding_method = "bert"
    tutorial = dataset[0]
    segment = None
    info_type = "Supplementary"
    n = 5 ### number of IUs to retrieve

    response = context_similarity_retrieval(embedding_method, task, dataset, tutorial, segment, info_type, n)
    print(json.dumps(response, indent=4))

def run_rag():
    pass

def run_vanilla():
    pass

def main():
    task = MUFFIN_TASK
    # task = CUSTOM_TASKS[14]
    # task = CROSS_TASK_TASKS[0]
    # task = CROSS_TASK_TASKS[1]
    # task = CUSTOM_TASKS[13]
    print(f"Running for task: {task}")

    #run_context_similarity(task)
    run_framework(task)

if __name__ == "__main__":
    main()