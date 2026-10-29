"""Convenient entry point; works from any working directory."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "task_rep"))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"demo", "export"}:
        command = sys.argv.pop(1)
        runpy.run_module("demo" if command == "demo" else "export_bundle", run_name="__main__")
    elif len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python main.py {demo,export,list,run,show,compare,budget,tree,dump} ...")
        print("  demo     Run a synthetic example without API keys or network access")
        print("  export   Export a saved run and knowledge base for the interfaces")
        print("  run      Run a configured pipeline experiment (uses model APIs)")
        print("Use python main.py COMMAND --help for command options.")
    else:
        runpy.run_module("pipeline.cli", run_name="__main__")
