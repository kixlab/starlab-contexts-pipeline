from pydantic_models.evaluation import MetricScale
from typing import Optional

from helpers.dataset import CROSS_TASK_TASKS, BIG_CUSTOM_TASKS, CUSTOM_TASKS

from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

from eval.llm_judge import relevance_absolute_evaluation, relevance_comparative_evaluation


TECH_EVAL_PATH = "./static/results/tech_eval/"
RESPONSE_ITEMS_CAP = 10 ### explicit cap on the number of response items

class DatasetConfig():
    label: str
    tasks: list[str]
    sample_per_task: int
    query: str

    def __init__(self, label, tasks, sample_per_task, query):
        self.label = label
        self.tasks = tasks
        self.sample_per_task = sample_per_task
        self.query = query
        if label is None:
            raise ValueError(f"label must be set, but got {label}")
        if tasks is None:
            raise ValueError(f"tasks must be set, but got {tasks}")
        if sample_per_task is None:
            raise ValueError(f"sample_per_task must be set, but got {sample_per_task}")
        if sample_per_task < 1:
            raise ValueError(f"sample_per_task must be at least 1, but got {sample_per_task}")
        if query is None:
            raise ValueError(f"query must be set, but got {query}")
        if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"tasks must be a list of strings, but got {tasks}")
        if len(tasks) == 0:
            raise ValueError(f"tasks must be a non-empty list, but got {tasks}")
    
    def __str__(self):
        return f"DatasetConfig(label={self.label}, tasks={self.tasks}, sample_per_task={self.sample_per_task}, query={self.query})"
    
    def to_dict(self):
        return {
            "label": self.label,
            "tasks": self.tasks,
            "sample_per_task": self.sample_per_task,
            "query": self.query,
        }

    def from_dict(obj):
        return DatasetConfig(
            label=obj["label"],
            tasks=obj["tasks"],
            sample_per_task=obj["sample_per_task"],
            query=obj["query"],
        )

class MethodConfig():
    label: str
    embedding_method: Optional[str]
    doc_k: Optional[int]
    doc_score_threshold: Optional[float]
    func: callable
    version: Optional[str]
    response_score_threshold: Optional[float]
    gen_model: Optional[str]
    
    def __init__(self, label, embedding_method, doc_k, doc_score_threshold, func, version, response_score_threshold, gen_model):
        self.label = label
        self.embedding_method = embedding_method
        self.doc_k = doc_k
        self.doc_score_threshold = doc_score_threshold
        self.func = func
        self.version = version
        self.response_score_threshold = response_score_threshold
        self.gen_model = gen_model
        if doc_k is not None and doc_k < 1:
            raise ValueError(f"doc_k must be at least 1, but got {doc_k}. If you want to use all retrieved documents, set doc_k to None.")
        if response_score_threshold is not None and (response_score_threshold < 0 or response_score_threshold > 1):
            raise ValueError(f"response_score_threshold must be between 0 and 1, but got {response_score_threshold}. If you want to retrieve all information pieces, set response_score_threshold to 0.")
        if embedding_method is None or (embedding_method != "tfidf" and embedding_method != "openai" and embedding_method != "bert"):
            raise ValueError(f"embedding_method must be one of 'tfidf', 'openai', or 'bert', but got {embedding_method}")
        if func is None:
            raise ValueError(f"func must be set, but got {func}")
        if label is None:
            raise ValueError(f"label must be set, but got {label}")

    def __str__(self):
        return f"MethodConfig(label={self.label}, embedding_method={self.embedding_method}, doc_k={self.doc_k}, doc_score_threshold={self.doc_score_threshold}, func={self.func.__name__}, version={self.version}, response_score_threshold={self.response_score_threshold}, gen_model={self.gen_model})"

    def to_dict(self):
        return {
            "label": self.label,
            "embedding_method": self.embedding_method,
            "doc_k": self.doc_k,
            "doc_score_threshold": self.doc_score_threshold,
            "func": "function",
            "version": self.version,
            "response_score_threshold": self.response_score_threshold,
            "gen_model": self.gen_model,
        }


class EvalConfig():
    label: str
    func: callable
    metric: MetricScale
    judge_model: Optional[str]
    n: Optional[int]
    
    def __init__(self, label, func, metric, judge_model, n):
        self.label = label
        self.func = func
        self.metric = metric
        self.judge_model = judge_model
        self.n = n
        
        if label is None:
            raise ValueError(f"label must be set, but got {label}")
        if func is None:
            raise ValueError(f"func must be set, but got {func}")
        if metric is None or not isinstance(metric, MetricScale):
            raise ValueError(f"metric must be set and must be one of {MetricScale}, but got {metric}")
        if judge_model is None:
            raise ValueError(f"judge_model must be set, but got {judge_model}")
        if n is not None and (n < 1 or n > RESPONSE_ITEMS_CAP):
            raise ValueError(f"n must be at least 1 and at most {RESPONSE_ITEMS_CAP}, but got {n}. If you want to evaluate the response 1-by-1, set n to None. Does not work for comparative evaluation.")
    
    def __str__(self):
        return f"EvalConfig(label={self.label}, func={self.func.__name__}, metric={self.metric}, judge_model={self.judge_model}, n={self.n})"

    def to_dict(self):
        return {
            "label": self.label,
            "func": self.func.__name__,
            "metric": self.metric,
            "judge_model": self.judge_model,
            "n": self.n,
        }


QUERIES = {
    "full": "Given a tutorial, identify and retrieve relevant `{info_type}` information that is missing from the tutorial, and arrange the results by relevance. The following defines `{info_type}` information:\n{info_type_description}",
    "segment": "Given a tutorial and a highlighted segment, identify and retrieve relevant `{info_type}` information that is missing from the segment, and arrange the results by relevance. The following defines `{info_type}` information:\n{info_type_description}",
}

### naming convention: {dataset_name}_{query_key}_{sample_per_task}_n{n}
DATASETS = {
    "test_full_10": DatasetConfig(
        label="test_full_10",
        tasks=[BIG_CUSTOM_TASKS[0]],
        sample_per_task=10,
        query=QUERIES["full"],
    ),
    "test_segment_10": DatasetConfig(
        label="test_segment_10",
        tasks=[CROSS_TASK_TASKS[0]],
        sample_per_task=10,
        query=QUERIES["segment"],
    ),
    "custom_full_50": DatasetConfig(
        label="custom_full_50",
        tasks=BIG_CUSTOM_TASKS,
        sample_per_task=50,
        query=QUERIES["full"],
    ),
    "cross_full_50": DatasetConfig(
        label="cross_full_50",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=50,
        query=QUERIES["full"],
    ),
    "cross_segment_10": DatasetConfig(
        label="cross_segment_10",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=50,
        query=QUERIES["segment"],
    ),
}

### naming convention: {method_name}_{embedding_method}_t{response_score_threshold?}_v{version?}_k{k?}_{gen_model?}
METHODS = {
    ### RAG GPT-4.1-mini
    "vanilla_tfidf_gpt-4-1-mini": MethodConfig(
        label="vanilla_tfidf_gpt-4-1-mini",
        embedding_method="tfidf",
        doc_k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_tfidf_k2_gpt-4-1-mini": MethodConfig(
        label="RAG_tfidf_k2_gpt-4-1-mini",
        embedding_method="tfidf",
        doc_k=2,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_tfidf_k4_gpt-4-1-mini": MethodConfig(
        label="RAG_tfidf_k4_gpt-4-1-mini",
        embedding_method="tfidf",
        doc_k=4,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_tfidf_k8_gpt-4-1-mini": MethodConfig(
        label="RAG_tfidf_k8_gpt-4-1-mini",
        embedding_method="tfidf",
        doc_k=8,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_tfidf_k16_gpt-4-1-mini": MethodConfig(
        label="RAG_tfidf_k16_gpt-4-1-mini",
        embedding_method="tfidf",
        doc_k=16,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_tfidf_k32_gpt-4-1-mini": MethodConfig(
        label="RAG_tfidf_k32_gpt-4-1-mini",
        embedding_method="tfidf",
        doc_k=32,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    
    ### OURS
    "ours_similarity_openai_t05_vlatest": MethodConfig(
        label="ours_similarity_openai_t05_vlatest",
        embedding_method="openai",
        doc_k=None,
        doc_score_threshold=None,
        func=generic_call_context_similarity,
        version=None,
        response_score_threshold=0.5,
        gen_model=None,
    ),
    "ours_distance_openai_t05_vlatest": MethodConfig(
        label="ours_distance_openai_t05_vlatest",
        embedding_method="openai",
        doc_k=None,
        doc_score_threshold=None,
        func=generic_call_shortest_path,
        version=None,
        response_score_threshold=0.5,
        gen_model=None,
    ),

    ### RAG GPT-5-mini
    "vanilla_tfidf_gpt-5-mini": MethodConfig(
        label="vanilla_tfidf_gpt-5-mini",
        embedding_method="tfidf",
        doc_k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_tfidf_k2_gpt-5-mini": MethodConfig(
        label="RAG_tfidf_k2_gpt-5-mini",
        embedding_method="tfidf",
        doc_k=2,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_tfidf_k4_gpt-5-mini": MethodConfig(
        label="RAG_tfidf_k4_gpt-5-mini",
        embedding_method="tfidf",
        doc_k=4,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_tfidf_k8_gpt-5-mini": MethodConfig(
        label="RAG_tfidf_k8_gpt-5-mini",
        embedding_method="tfidf",
        doc_k=8,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_tfidf_k16_gpt-5-mini": MethodConfig(
        label="RAG_tfidf_k16_gpt-5-mini",
        embedding_method="tfidf",
        doc_k=16,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_tfidf_k32_gpt-5-mini": MethodConfig(
        label="RAG_tfidf_k32_gpt-5-mini",
        embedding_method="tfidf",
        doc_k=32,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        response_score_threshold=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
}

### naming convention: {criteria}_{joint/apiece}_{metric type}_{judge_model}_{scale}
EVALS = {
    "relevance_absolute_gpt-5-mini_binary_n": EvalConfig(
        label="relevance_absolute_gpt-5-mini_binary_n",
        func=relevance_absolute_evaluation,
        metric=MetricScale.BINARY,
        judge_model="gpt-5-mini-2025-08-07",
        n=None,
    ),
    "relevance_absolute_gpt-5-mini_likert-3_n": EvalConfig(
        label="relevance_absolute_gpt-5-mini_likert-3_n",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_3,
        judge_model="gpt-5-mini-2025-08-07",
        n=None,
    ),
    "relevance_absolute_gpt-5-mini_likert-5_n": EvalConfig(
        label="relevance_absolute_gpt-5-mini_likert-5_n",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_5,
        judge_model="gpt-5-mini-2025-08-07",
        n=None,
    ),
    "relevance_absolute_gpt-5-mini_likert-5_n5": EvalConfig(
        label="relevance_absolute_gpt-5-mini_likert-5_n5",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_5,
        judge_model="gpt-5-mini-2025-08-07",
        n=5,
    ),
    "relevance_comparative_gpt-5-mini_comparison_n5": EvalConfig(
        label="relevance_comparative_gpt-5-mini_comparison_n5",
        func=relevance_comparative_evaluation,
        metric=MetricScale.COMPARISON,
        judge_model="gpt-5-mini-2025-08-07",
        n=5,
    ),
}