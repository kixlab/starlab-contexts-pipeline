# Optional video preparation

You can supply transcript JSON directly; see `../../examples/demo-dataset.json`.
No downloader or GPU is needed for the pipeline or its offline demo.

For video preparation, install `requirements-video.txt` from the repository
root, plus FFmpeg and Node.js on your operating system. Add your own task and
video IDs to `data/tasks.json`, then run from `task_rep/`:

```bash
python -m dataset --help
python -m dataset list
python -m dataset info YOUR_TASK_ID
python -m dataset build YOUR_TASK_ID --fps 1.0
```

The default transcription uses OpenAI and requires your exported
`OPENAI_API_KEY`. `--transcript-source youtube` uses available YouTube subtitles
instead. Network access and source permissions are required for downloads.
Downloads, frames, transcripts, and task datasets go under the configured
`CIM_STORAGE_DIR` (default: `task_rep/static/`) and are ignored by Git.

If authentication is needed, set `YT_DLP_COOKIES_FILE` to a cookies file outside
the repo or `YT_DLP_COOKIES_FROM_BROWSER` to your browser. No cookies are included.
Video download behavior depends on the external site and is outside offline CI.
