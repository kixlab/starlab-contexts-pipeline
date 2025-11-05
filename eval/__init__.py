from pydantic_models.evaluation import MetricScale
from typing import Optional

class DatasetConfig():
    label: str
    tasks: list[str]
    sample_per_task: int
    query: int
    sample_segment_per_tutorial: int
    n: int

    def __init__(self, label, tasks, sample_per_task, query, n):
        self.label = label
        self.tasks = tasks
        self.sample_per_task = sample_per_task
        self.query = query
        self.sample_segment_per_tutorial = None
        self.n = n
    
    def __str__(self):
        return f"DatasetConfig(label={self.label}, tasks={self.tasks}, sample_per_task={self.sample_per_task}, query={self.query}, sample_segment_per_tutorial={self.sample_segment_per_tutorial}, n={self.n})"
    
    def to_json(self):
        return {
            "label": self.label,
            "tasks": self.tasks,
            "sample_per_task": self.sample_per_task,
            "query": self.query,
            "sample_segment_per_tutorial": self.sample_segment_per_tutorial,
            "n": self.n,
        }

class MethodConfig():
    label: str
    embedding_method: str
    k: int
    doc_score_threshold: float
    func: callable
    version: str
    gen_model: Optional[str]
    
    def __init__(self, label, embedding_method, k, doc_score_threshold, func, version, gen_model):
        self.label = label
        self.embedding_method = embedding_method
        self.k = k
        self.doc_score_threshold = doc_score_threshold
        self.func = func
        self.version = version
        self.gen_model = gen_model

    def __str__(self):
        return f"MethodConfig(label={self.label}, embedding_method={self.embedding_method}, k={self.k}, doc_score_threshold={self.doc_score_threshold}, func={self.func.__name__}, version={self.version}, gen_model={self.gen_model})"

    def to_json(self):
        return {
            "label": self.label,
            "embedding_method": self.embedding_method,
            "k": self.k,
            "doc_score_threshold": self.doc_score_threshold,
            "func": "function",
            "version": self.version,
            "gen_model": self.gen_model,
        }

class EvalConfig():
    label: str
    func: callable
    metric: MetricScale
    judge_model: Optional[str]
    
    def __init__(self, label, func, metric, judge_model):
        self.label = label
        self.func = func
        self.metric = metric
        self.judge_model = judge_model
    
    def __str__(self):
        return f"EvalConfig(label={self.label}, func={self.func.__name__}, metric={self.metric}, judge_model={self.judge_model})"

    def to_json(self):
        return {
            "label": self.label,
            "func": "function",
            "metric": self.metric,
            "judge_model": self.judge_model,
        }