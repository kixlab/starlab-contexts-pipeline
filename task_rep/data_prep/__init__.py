"""Dataset utilities; video dependencies load only when requested."""
from .task_manager import Task, TaskManager
from .narration_checker import check_narration_quality, NarrationCheckResult

def __getattr__(name):
    if name == "VideoDownloader":
        from .video_downloader import VideoDownloader
        return VideoDownloader
    if name == "build_task_dataset":
        from .task_dataset_builder import build_task_dataset
        return build_task_dataset
    raise AttributeError(name)
