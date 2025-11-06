import json
import os
import traceback
from src.framework_split import construct_cim_split
from src.framework_iter import construct_cim_iter
from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS, BIG_CUSTOM_TASKS

from src.cim_methods import context_similarity_retrieval

from helpers.dataset import get_dataset

from helpers.cim_scripts import FRAMEWORK_PATH

import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr

def run_framework(args):
    """
    Runs one task, writing that task's stdout+stderr to a task-specific file.
    Returns a small, picklable status dict (never raises).
    """
    task = args["task"]
    version = args["version"]

    output_dir = os.path.join(FRAMEWORK_PATH, version)
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"split_{task.replace(' ', '_').lower()}.txt"
    output_path = os.path.join(output_dir, output_file)

    status = {"task": task, "version": version, "ok": True, "error": None}

    try:
        # open once per task; capture BOTH stdout and stderr
        with open(output_path, "w", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
            dataset = get_dataset(task)  # your function
            _ = construct_cim_split(task, dataset, version)  # your function
            print("COMPLETED", task, flush=True)
    except Exception as e:
        # Never let third-party exceptions cross the process boundary
        status["ok"] = False
        status["error"] = {
            "etype": type(e).__name__,
            "msg": str(e),
            "trace": traceback.format_exc(),
            "log_file": output_path,
        }
        # Also write the traceback into the task's log file
        try:
            with open(output_path, "a") as f:
                f.write("\n[EXCEPTION]\n")
                f.write(status["error"]["trace"])
                f.flush()
        except Exception:
            # best-effort; ignore logging failures
            pass

    return status

def parallelize(func, args, num_workers=10):
    """
    Stream results to avoid buffer backpressure; never raise from workers.
    """
    results = []
    # maxtasksperchild is handy if tasks are heavy and may leak memory
    with mp.Pool(processes=num_workers, maxtasksperchild=100) as pool:
        for res in pool.imap_unordered(func, args, chunksize=1):
            results.append(res)
    return results

def run_framework_parallel(args):
    results = parallelize(run_framework, args, num_workers=5)

    # Surface any failures immediately and point to their log files
    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"[SUMMARY] {len(failures)} task(s) failed:")
        for r in failures:
            err = r["error"]
            print(f" - {r['task']}: {err['etype']}: {err['msg']}\n    see: {err['log_file']}")
    else:
        print("COMPLETED FULL RUN!")

def run_framework_big_custom_tasks(version):
    args = [{"task": task, "version": version} for task in BIG_CUSTOM_TASKS]
    run_framework_parallel(args)

def run_framework_cross_tasks(version):
    args = [{"task": task, "version": version} for task in CROSS_TASK_TASKS]
    run_framework_parallel(args)

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
    mp.set_start_method("spawn", force=True)
    # run_framework_custom_tasks("full_run_2")
    # run_framework_big_custom_tasks("full_run_4")
    # run_framework({"task": BIG_CUSTOM_TASKS[0], "version": "full_run_2"})
    run_framework_cross_tasks("full_run_5")
    # run_framework({"task": CROSS_TASK_TASKS[0], "version": "full_run_3"})