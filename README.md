# TaskBinge Multi-Video Processing Pipeline

This is the repository for the pipeline for TaskBinge.

## Repository Structure

- [`helpers`](/helpers/): Helper scripts
- [`src`](/src/): Video and VideoPool Class implementations
- [`pydantic_models`](/pydantic_models/): PyDantic models used for parsing LM responses
- [`static`](/static/): Storage of processing results. 
- [`preprocess.py`](/preprocess.py): Script that runs single processing of the pipeline for a particular task.
- [`metadata.json`](/metadata.json/): JSON-file with all the task data (title, video links).
- [`environment.yml`](/environment.yml): Installation packages.
- [`README.md`](/README.md): Instructions file.

## Development environment

-   Ubuntu 18.04, CUDA 12.1

## Installation

1. Create a new [conda](https://docs.conda.io/en/latest/) environment (Python 3.10)

```bash
conda env create -f environment.yml
conda activate taskbinge
```

2. Install [CLIP](https://github.com/openai/CLIP) package.
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Run

Run a single instance of the pipeline for one of the tasks in [`metadata`](/metadata/). There are two available tasks [`carbonara`] (How to cook carbonara pasta?) and [`remove-object`] (How to remove object with Photoshop?). You can add a separate entry for a custom task in 
```bash
# -h
#                       Help
# -t TASKID, --task TASKID
#                       The task-id. Ex: carbonara
python preprocess.py [-t TASKID]
```

For example:
```bash
python preprocess.py -t carbonara
```

Notes:
- Export environment variable `$OPENAI_API_KEY` with appropriate OPENAI_API_KEY from [OpenAI](https://openai.com/).

## Output

Processing results are stored in under `static/results/{task-id}/ouptut.json`.

- Output format for notables/hooks is as follows:
```python
{
    "id": str,
    "video_id": str,
    "subgoal": str,
    "aspect": str,
    "relation": str,
    "title": str,
    "description": str,
    "comparison": str,
    "importance": float,
    "links": [
        {
            "id": str,
            "title": str,
            "description": str,
            "reasoning": str,
            "comparison": str,
            "subgoal": str,
            "aspect": str,
            "relation": str,
            "other_video_id": str,
            "importance": float,
            "uniqueness": float,
            "other_seconds": float
        }
    ]
}
```

For any questions please contact: [Bekzat Tilekbay](mailto:tlekbay.b@gmail.com)