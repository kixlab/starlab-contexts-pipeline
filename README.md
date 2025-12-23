# Context Information Model (CIM) Framework Construction

This repository contains the pipeline for constructing Context Information Models (CIM) from instructional video datasets. The system processes video transcripts to extract information units, build facet candidates, and generate context schemas that organize knowledge in a structured, fine-grained manner.

## Repository Structure

- [`_main.py`](_main.py): Main entry point for constructing CIM frameworks
- [`src`](src/): Core framework construction logic
  - [`framework_split.py`](src/framework_split.py): Main pipeline for processing videos and building CIM schemas
- [`helpers`](helpers/): Helper scripts and utilities
  - [`cim_scripts.py`](helpers/cim_scripts.py): CIM framework construction functions
  - [`dataset.py`](helpers/dataset.py): Dataset loading and task definitions
  - [`nlp.py`](helpers/nlp.py): Natural language processing utilities
  - [`pydantic_models`](pydantic_models/): Pydantic models for data validation
  - [`framework.py`](pydantic_models/framework.py): Framework-related models
  - [`evaluation.py`](pydantic_models/evaluation.py): Evaluation models
  - [`rag.py`](pydantic_models/rag.py): RAG-related models
  - [`summarization.py`](pydantic_models/summarization.py): Summarization models
- [`prompts`](prompts/): Prompt templates for LLM interactions
  - [`framework.py`](prompts/framework.py): Framework construction prompts
- [`static`](static/): Storage of datasets and processing results
  - [`datasets`](static/datasets/): Input video datasets in JSON format
  - [`results`](static/results/): Processing results and generated CIM schemas
- [`requirements.txt`](requirements.txt): Python package dependencies

## Development environment

-   Ubuntu 22.04, CUDA 12.8

## Installation

1. Create a new [conda](https://docs.conda.io/en/latest/) environment (Python 3.10)

```bash
conda create -n cim python=3.10
conda activate cim
```

2. Install required dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables for LMs (OpenAI)

Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys). You'll need to create an account and generate an API key.

**Option A: Temporary (current terminal session only)**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Conda environment variable**

If you're using conda, you can set it as an environment variable:

```bash
# Make sure conda environment is activate (i.e., `cim`)
conda env config vars set OPENAI_API_KEY="your-api-key-here"
conda activate cim  # Reactivate the environment
```

**Verify the environment variable is set:**

```bash
echo $OPENAI_API_KEY
```

This should output your API key. If it's empty, the environment variable is not set correctly.

## Available Tasks

The system supports multiple instructional video tasks. Available tasks include:

**Cross-Task Tasks:**
- Change a Tire
- Build Simple Floating Shelves
- Make French Toast
- Make Irish Coffee

**Custom Tasks:**
- How to Make a Sushi Roll
- How to Make Caramel Apples
- How to Make a Milkshake Without Ice Cream
- How to Grill Steak
- How to Make Scrambled Eggs in a Microwave
- How to Grow Hydrangea from Cuttings
- How to Grow a Pumpkin
- How to Clean Bathroom Tile
- How to Polish Stainless Steel
- How to Clean a Glass Top Stove
- How to Get Rid of a Wasp's Nest
- How to Plant a Living Christmas Tree
- How to Wrap Your Hands for Boxing
- How to Catch Trout
- How to Make a Paper Hat

## Run

Run the pipeline to construct a CIM framework for a specific task:

```bash
# -t TASK, --task TASK
#                       The task to construct the CIM for (required)
# -e EMBEDDING_METHOD, --embedding_method EMBEDDING_METHOD
#                       Embedding method: openai (default) | bert
# -x EXTRACTION_MODEL, --extraction_model EXTRACTION_MODEL
#                       LLM used for information unit extraction (default: gpt-4.1-mini-2025-04-14)
# -g GENERATION_MODEL, --generation_model GENERATION_MODEL
#                       LLM used for facet mining and labeling (default: gpt-4.1-mini-2025-04-14)
# -v VERSION, --version VERSION
#                       Optionally specify the version of the CIM (default: full_run_1)
python _main.py -t <task> [-e <embedding_method>] [-x <extraction_model>] [-g <generation_model>] [-v <version>]
```

Examples:
```bash
# Minimal (uses defaults: openai embeddings, gpt-4.1-mini-2025-04-14 models)
python _main.py -t "How to Make a Paper Hat"

# Specify version label for results directory
python _main.py -t "How to Make a Paper Hat" -v "full_run_1"

# Use local BERT embeddings instead of OpenAI embeddings
python _main.py -t "How to Make a Paper Hat" -e bert

# Override LLMs used for extraction/generation
python _main.py -t "How to Make a Paper Hat" -x gpt-4.1-mini-2025-04-14 -g gpt-4.1-mini-2025-04-14
```

The pipeline will:
1. Load the dataset for the specified task from `static/datasets/`
2. Build information units from video transcripts
3. Mine facet candidates iteratively
4. Compute the Pareto frontier using knapsack optimization
5. Generate the final context schema

## Output

Processing results are stored in `static/results/{task-name}/split_results_{version}.json`.

The output format includes:

```python
{
    "context_schema": [
        {
            "id": str,
            "type": str,  # e.g., "what", "why", "when", "where"
            "title": str,
            "title_plural": str,
            "definition": str,
            "guidelines": [str, ...],
            "vocabulary": [
                {
                    "label": str,
                    "definition": str,
                },
                ...
            ]
        },
        ...
    ],
    "facet_candidates": [...],  # All discovered facet candidates
    "labeled_dataset": [...]     # Processed dataset with labels
}
```

The context schema organizes knowledge into facets (dimensions of variation) with associated vocabularies (labels) that can be used to partition and structure instructional content.

## What is CIM?
We introduce Context-Information Maps (CIM) to automatically organize scattered task knowledge across a large corpus of tutorial videos about the same task. By transforming static tutorials into a unified bipartite graph that links specific "contexts" (applicability conditions such as tools, ingredients, or goals) with atomic "information units" (specific instructions, tips, or explanations), the system explicitly aligns diverse knowledge pieces from different sources according to when/where they are relevant (i.e., task contexts). CIM enables applications such as the automatic augmentation of tutorials with missing information, navigation between alternative methods across different sources, and the analytical detection of knowledge gaps or saturation points within a corpus of tutorials for the task.

## Notes

- The system uses OpenAI embeddings (`text-embedding-3-large`) and language models (`gpt-4.1-mini-2025-04-14`) for extraction and processing
- Processing is iterative and may take time depending on dataset size
- Intermediate results are saved automatically, allowing for resumption of interrupted runs
- The system targets a discriminativeness score below 0.8 to ensure fine-grained knowledge organization

For any questions please contact: [Bekzat Tilekbay](mailto:tilekbay@kaist.ac.kr)
