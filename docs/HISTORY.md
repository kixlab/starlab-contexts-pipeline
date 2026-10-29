# Development history import

This distribution combines filtered histories rather than flattening copied
code into a single initial commit:

| Source | Retained development commits | Destination |
| --- | ---: | --- |
| Earlier `starlab-contexts-pipeline` | 94 | `legacy/` (moved in an integration commit) |
| `probe/task_rep` | 112 | `task_rep/` |
| `cim-tutorial-analysis` inspector and graph | 56 | `explorer/` |

Two merge commits connect these histories, followed by integration work for
portable paths, public examples, exports, documentation, and checks. Authors,
authored dates, messages, and relevant changes are retained. No empty commits,
invented authors, or fabricated dates were introduced.

Filtering excludes unrelated source files, the two experimental agents
(`agentic/`, `design_agent/`), credentials, private study configuration and
responses, raw research datasets, generated model results, caches, cookies,
notebooks, and machine-specific logs. Existing history contained credential-like
strings outside the retained code paths. The filtered graph is scanned across
all reachable file blobs, as well as the final tracked tree.

[`history/`](history/) contains original-to-filtered commit maps. A zero target
hash means a commit had no retained changes. Hashes change when file paths,
trees, or ancestry change; preserving every original SHA would also preserve
excluded history. The maps keep source commits traceable without shipping that
history. Private source repos and their uncommitted work were read only.

This is a local reorganization. It does not push changes, alter GitHub settings,
or replace remote branches. Publishing this sanitized graph over an existing
remote branch requires an intentional history replacement, and older clones
retain their original contents. The original local pipeline is backed up
outside the public repo. Any real credentials previously committed need to be
revoked at their provider; rewriting history does not revoke them.

Commit totals describe reachable Git history. They do not guarantee a particular
contribution count or eligibility under an external threshold's rules.
