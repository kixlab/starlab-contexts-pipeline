import json
import os
import traceback
from src.framework_split import construct_cim_split
from src.framework_iter import construct_cim_iter
from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS, BIG_CUSTOM_TASKS

from src.cim_methods import context_similarity_retrieval

from helpers.dataset import get_dataset

from helpers.cim_scripts import FRAMEWORK_PATH

from contextlib import redirect_stdout, redirect_stderr

def run_framework(task, version):
    output_dir = os.path.join(FRAMEWORK_PATH, version)
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"split_{task.replace(' ', '_').lower()}.txt"
    output_path = os.path.join(output_dir, output_file)

    # open once per task; capture BOTH stdout and stderr
    with open(output_path, "w", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        dataset = get_dataset(task)  # your function
        _ = construct_cim_split(task, dataset, version)  # your function
        print("COMPLETED", task)

def run_framework_small_custom_tasks(version):
    SMALL_CUSTOM_TASKS = []
    for task in CUSTOM_TASKS:
        if task in BIG_CUSTOM_TASKS:
            continue
        SMALL_CUSTOM_TASKS.append(task)
    for task in SMALL_CUSTOM_TASKS:
        run_framework(task, version)
    print("COMPLETED FULL RUN SMALL CUSTOM TASKS!")

def run_framework_big_custom_tasks(version):
    for task in BIG_CUSTOM_TASKS:
        run_framework(task, version)
    print("COMPLETED FULL RUN BIG CUSTOM TASKS!")

def run_framework_cross_tasks(version):
    for task in CROSS_TASK_TASKS:
        run_framework(task, version)
    print("COMPLETED FULL RUN CROSS TASK TASKS!")

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
    version = "trial_1"
    task = MUFFIN_TASK
    # task = CUSTOM_TASKS[14]
    # task = CROSS_TASK_TASKS[0]
    # task = CROSS_TASK_TASKS[1]
    # task = CUSTOM_TASKS[13]
    print(f"Running for task: {task}")

    #run_context_similarity(task)
    run_framework(task, version)

if __name__ == "__main__":
    # run_framework(MUFFIN_TASK, "full_run_0")
    
    # run_framework_small_custom_tasks("full_run_0")
    run_framework(BIG_CUSTOM_TASKS[0], "full_run_8")
    # run_framework_big_custom_tasks("full_run_5")
    # run_framework({"task": BIG_CUSTOM_TASKS[0], "version": "full_run_2"})
    # run_framework_cross_tasks("full_run_4")
    # run_framework({"task": CROSS_TASK_TASKS[0], "version": "full_run_3"})