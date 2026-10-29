# Earlier CIM pipeline

This directory preserves the earlier framework-split implementation. The current
pipeline and public examples are documented in the [root README](../README.md).
The two implementations have separate dependencies; use a separate virtual
environment if you need to reproduce an earlier experiment.

From this directory, install `requirements.txt`, export your own
`OPENAI_API_KEY`, and put an authorized legacy-format transcript dataset at
`static/datasets/<task_name_in_lowercase_with_underscores>_framework_raw.json`.
The former research datasets and generated results are excluded from this public
distribution. Then run:

```bash
python main.py -t "Your Task Name" -v my_run
```

This implementation calls model APIs; optional BERT embeddings also install
PyTorch through sentence-transformers. Results go to `static/results/` and model
batch logs to `static/results/lm-batches/`, both ignored by Git. For a small,
portable example that needs no credentials, use `python main.py demo` from the
repository root instead. Automated verification covers the current pipeline;
live legacy model runs are not exercised in CI.
