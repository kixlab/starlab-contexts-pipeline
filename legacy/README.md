# Framework-split CIM construction

This implementation constructs context schemas from legacy-format tutorial
datasets. The [root README](../README.md) describes the task-representation
pipeline and its offline example. Use a separate virtual environment for the
dependencies listed in this directory's `requirements.txt`.

From this directory, install `requirements.txt`, export your own
`OPENAI_API_KEY`, and place a transcript dataset at
`static/datasets/<task_name_in_lowercase_with_underscores>_framework_raw.json`.
Then run:

```bash
python main.py -t "Your Task Name" -v my_run
```

Model-backed construction calls external APIs. Optional BERT embeddings use
PyTorch through sentence-transformers. Results go to `static/results/` and
model batch logs to `static/results/lm-batches/`, both ignored by Git.

For a small example that needs no credentials, run `python main.py demo` from
the repository root. CI covers the task-representation pipeline, offline
examples, shared bundle exports, and web explorer; live framework-split model
runs require your own data and credentials.
