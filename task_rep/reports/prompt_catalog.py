"""Render the public batch and iterative proposal prompt templates (offline)."""
from pathlib import Path
from facets.prompts.fa_facet_init import USER_PROMPT
from facets.prompts.fa_facet_iterative import _INSTRUCTIONS

def main():
    path = Path(__file__).resolve().parents[1] / "docs" / "prompt_catalog.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Proposal prompt templates\n\n## Batch\n\n```text\n" + USER_PROMPT
                    + "\n```\n\n## Iterative\n\n```text\n" + _INSTRUCTIONS + "\n```\n")
    print(path)

if __name__ == "__main__":
    main()
