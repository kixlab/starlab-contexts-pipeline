import sys
import argparse

from src.framework_split import construct_cim_split
from helpers.dataset import CUSTOM_TASKS, CROSS_TASK_TASKS

from helpers.dataset import get_dataset

def parse_args(args):
    """
    Help:
    python _main.py --task <task> --version <version>
    python _main.py -t <task> -v <version>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True, help=f"The task to construct the CIM for (e.g., {CUSTOM_TASKS[14]})")
    parser.add_argument("-v", "--version", type=str, required=False, default="full_run_1", help=f"Optionally specify the version of the CIM (e.g., full_run_1)")
    parser.add_argument("-e", "--embedding_method", type=str, required=False, default="openai", help=f"Optionally specify the embedding method (e.g., openai, bert)")
    parser.add_argument("-x", "--extraction_model", type=str, required=False, default="gpt-4.1-mini-2025-04-14", help=f"Optionally specify the extraction model (e.g., gpt-4.1-mini-2025-04-14)")
    parser.add_argument("-g", "--generation_model", type=str, required=False, default="gpt-4.1-mini-2025-04-14", help=f"Optionally specify the generation model (e.g., gpt-4.1-mini-2025-04-14)")
    return parser.parse_args(args)

def construct_cim(task, embedding_method, extraction_model, generation_model, version):
    dataset = get_dataset(task)
    results = construct_cim_split(task, dataset, embedding_method, extraction_model, generation_model, version)
    schema = results["context_schema"]
    dataset = results["labeled_dataset"]
    return schema, dataset

def main(task, embedding_method, extraction_model, generation_model, version):
    if task is None:
        raise ValueError(f"Task is required!")
    schema, _ = construct_cim(task, embedding_method, extraction_model, generation_model, version)
    for facet in schema:
        print(f"`{facet['title']}`: {facet['definition']}")
        for label in facet["vocabulary"]:
            print(f"\t`{label['label']}`: {label['definition']}")

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    main(args.task, args.embedding_method, args.extraction_model, args.generation_model, args.version)