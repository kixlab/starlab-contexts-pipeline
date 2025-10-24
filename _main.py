from src.framework_v0 import construct_cim
from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS

def run_framework(task):
    dummy = "segmentation_v1"
    _ = construct_cim(task, dummy)
    print("COMPLETED!")

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

    run_framework(task)


if __name__ == "__main__":
    main()