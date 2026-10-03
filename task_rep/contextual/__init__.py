"""
Shared core for contextual-unit runners (``pipeline`` and ``agentic``).

- ``settings``: RunSettings (shared config) and seed derivation
- ``lifecycle``: candidate lifecycle (dedup, IRR, vocabulary, labeling)
- ``evaluation``: schema evaluation and pool selection
- ``models``: run state, candidate ledger, results, stop reasons
- ``store``: fingerprinted checkpoints and results
- ``fingerprint``: prompt-fingerprint fixture and hashing
- ``kb``: knowledge-base access, result lookup, model-backed services
- ``reporting``: summaries, ledger budget analysis, tau sweep, CSV export
- ``fakes``: deterministic lifecycle services for tests and offline runs

Depends only on the lower layers (facets, optimization, knowledge_items,
utils, construct); never on a runner package.
"""
