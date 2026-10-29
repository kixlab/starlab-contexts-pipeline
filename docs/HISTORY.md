# Component provenance

The system contains three component families:

| Component | Location | Responsibility |
| --- | --- | --- |
| Legacy framework construction | `legacy/` | Framework-split knowledge extraction and contextual-unit construction |
| Discovery Pipeline | `task_rep/` | Knowledge items and units, contextual units, canonical values, annotations, and evaluation |
| Design-space explorer | `explorer/` | Compare candidate organizations and explore the contextual-unit refinement DAG |

The task-representation modules share their development lineage with
`probe/task_rep`. The explorer shares its interface foundations with
`cim-tutorial-analysis`. Source authorship and implementation changes are
recorded in Git.

The files in [`history/`](history/) map component source commits to their
repository identifiers. A zero target identifier denotes a source change
outside the component's retained scope. These references support tracing a
module or behavior back to its implementation.

Runtime datasets, model request logs, credentials, downloaded media, and local
configuration belong in ignored storage. `scripts/check_public.py --history`
checks tracked files and reachable source history for common credential
patterns and excluded paths. Synthetic sample data and media are documented
separately alongside the examples.
