import sys
import os

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS
from analysis.frontier import plot_frontiers_facets, plot_frontiers_labels
from analysis.frontier import get_available_results, classify_facet_candidates

from analysis.display import show_task_stats

from analysis import ANALYSIS_PATH

def plot_results(results, piece_types, output_folder, classify=True):
    elbow_d = None
    y_axis = "discriminativeness"
    ### classify the facet candidates into common vs unique to the task
    if classify:
        sim_thresh = 0.9
        embedding_method = "openai"
        results = classify_facet_candidates(results, sim_thresh, embedding_method)
    
    plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder)
    plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder)

def main(folder):

    output_folder = os.path.join(ANALYSIS_PATH, folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    piece_types = ["Method - Subgoal", "Method - Instruction", "Method - Tool",
        "Supplementary - Tip", "Supplementary - Warning",
        "Explanation - Justification", "Explanation - Effect",
        "Description - Status", "Description - Context", "Description - Tool Specification",
        "Conclusion - Outcome", "Conclusion - Reflection",
    ]

    # tasks = CUSTOM_TASKS + [MUFFIN_TASK]

    # hat_tasks = [CUSTOM_TASKS[14], CUSTOM_TASKS[14]]
    # hat_dummies = ["trial_1", "full_run_1"]
    # results = get_available_results(hat_tasks, hat_dummies)

    tasks_trial_1 = [CUSTOM_TASKS[14]] + [MUFFIN_TASK]
    dummies_trial_1 = ["trial_1"] * 2
    # results = get_available_results(tasks_trial_1, dummies_trial_1)

    tasks_full_run_1 = CUSTOM_TASKS
    dummies_full_run_1 = ["full_run_1"] * len(tasks_full_run_1)

    tasks_full_run_2 = CUSTOM_TASKS
    dummies_full_run_2 = ["full_run_2"] * len(tasks_full_run_2)

    tasks_full_run_3 = CROSS_TASK_TASKS
    dummies_full_run_3 = ["full_run_3"] * len(tasks_full_run_3)

    tasks = tasks_full_run_1 + tasks_full_run_2 + tasks_full_run_3
    dummies = dummies_full_run_1 + dummies_full_run_2 + dummies_full_run_3

    results = get_available_results(tasks, dummies)
    # plot_results(results, piece_types, output_folder, classify=True)

    print("Tasks: ", len(results))
    for task, result in results.items():
        show_task_stats(task, result, piece_types)

if __name__ == "__main__":
    args = sys.argv[1:]
    folder = "all_frontier"
    if len(args) > 0:
        folder = args[0]
    main(folder)