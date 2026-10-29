"""Proposal: context components, prompt rendering, config validation, fingerprint."""

from __future__ import annotations

import pytest

from facets.prompts import fa_facet_init, fa_facet_labeling, fa_facet_vocabulary, formatters
from facets.prompts.fa_facet_iterative import (
    form_facet_iterative_request,
    process_facet_iterative_response,
)
from optimization import AcceptedSchemaSummary, RejectedSchemaSummary

from pipeline.config import (
    A1_CONTEXT,
    A2_CONTEXT,
    ContextComponent,
    PipelineConfig,
    ProposalStrategy,
)
from pipeline.fingerprint import prompt_fingerprint
from pipeline.proposal import BatchContext, HistoryEntry, IterativeContext, VLMFacetProposer

from tests.pipeline.helpers import label_book, make_config, make_facet, make_kb

SECTION_FACETS = "- Contextual factors in the pool:"
SECTION_ITEMS = "- Current labels of the tutorial segments"
SECTION_SCHEMAS = "- The current design space of factor combinations."
SECTION_HISTORY = "- Previously tried factors that are not in the pool:"
ALL_SECTIONS = {
    "facets": SECTION_FACETS,
    "items": SECTION_ITEMS,
    "schemas": SECTION_SCHEMAS,
    "history": SECTION_HISTORY,
}


def _text(request, index) -> str:
    content = request["messages"][index]["content"].content
    return "\n".join(b["text"] for b in content if b["type"] == "text")


def _request(components, *, pool=None, history=None, with_schemas=True, ids=None,
             max_schema_facets=None):
    kb = make_kb()
    pool = [make_facet("A"), make_facet("B")] if pool is None else pool
    ann = label_book([f.id for f in pool])
    accepted = [AcceptedSchemaSummary(id="A", facet_ids=["A"], size=2, D=0.1,
                                      eta2=1.0, H=1.0)] if (with_schemas and pool) else []
    rejected = [RejectedSchemaSummary(
        id="B", facet_ids=["B"], size=2, D=0.1, eta2=1.0, H=1.0,
        rejected_by_id="A", rejected_by_facet_ids=["A"], MI=0.9,
        refinement_score=0.9, refinement_threshold=0.2,
    )] if (with_schemas and pool) else []
    return form_facet_iterative_request(
        task="sushi",
        facets=pool,
        components=components,
        corpus=kb.by_video(),
        annotations=ann,
        knowledge_base=kb,
        accepted_schemas=accepted,
        rejected_schemas=rejected,
        history=history if history is not None else [
            HistoryEntry(title="Knife Type", scope="global", definition="Which knife is used.",
                         outcome="no usable values could be built from the tutorials"),
            HistoryEntry(title="Rice Kind", scope="global", definition="Kind of rice.",
                         outcome='duplicate of "Rice Type"'),
        ],
        ids=ids,
        max_schema_facets=max_schema_facets,
    )


def _render(components, **kwargs):
    return _text(_request(components, **kwargs), 1)


# ---------------------------------------------------------------------------
# Prompt rendering per component
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("components", [
    {"facets"}, {"history"}, {"facets", "history"}, {"facets", "items"},
    {"facets", "schemas"}, {"facets", "items", "schemas", "history"},
])
def test_each_component_renders_only_when_requested(components):
    text = _render(components)
    for name, header in ALL_SECTIONS.items():
        assert (header in text) == (name in components), name


@pytest.mark.parametrize("components", [{"items"}, {"schemas"}, {"items", "history"}])
def test_items_and_schemas_require_facets(components):
    with pytest.raises(ValueError, match="require facets"):
        _render(components)


def test_substring_context_is_rejected():
    with pytest.raises(TypeError):
        _render("facets+items")


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError):
        _render({"facets", "tree"})


def test_segment_text_is_only_in_the_system_message():
    systems = set()
    for components in ({"facets", "items", "history"}, {"facets", "schemas", "history"}):
        request = _request(components)
        assert "Segment i1" not in _text(request, 1)
        systems.add(_text(request, 0))
    batch = fa_facet_init.form_facet_init_request(task="sushi", videos=make_kb().by_video(),
                                                  num_facets=3)
    assert "Segment i1" not in _text(batch, 1)
    systems.add(_text(batch, 0))
    (system,) = systems  # a0, a1 and a2 share one system message
    assert '<segment id="S1">\nSegment i1\n</segment>' in system and "Video v2:" in system


def test_items_are_labels_by_segment_id():
    text = _render({"facets", "items"})
    assert "Video v1:\nS1: F1 = a; F2 = a\nS2: F1 = a; F2 = a\nS3: F1 = a; F2 = b" in text


def test_a1_empty_pool_first_iteration():
    text = _render({"facets", "items"}, pool=[], history=[])
    assert "No contextual factors in the pool yet." in text
    assert "No contextual factors in the pool yet, so no segment has labels." in text


def test_a2_empty_pool_first_iteration_has_no_data():
    text = _render({"facets", "schemas"}, pool=[], history=[])
    assert "Kept combinations:\n- none yet" in text
    assert "Pruned single factors:\n- none" in text


def test_a2_renders_schema_feedback_without_factor_details():
    text = _render({"facets", "schemas"})
    schemas = text[text.index(SECTION_SCHEMAS):]
    assert "Kept combinations:\n- C1: F1 (As), 2 cells" in schemas
    assert "- F2 (Bs): pruned; C1 (F1 (As)) already carries 90%" in schemas
    assert "<factor" not in schemas and "Last submission" not in schemas
    assert text.count('<factor id="F1">') == 1  # factor details once, in FACETS


def test_stable_ids_are_used_by_every_section():
    text = _render({"facets", "items", "schemas"}, ids={"A": "F2", "B": "F5"})
    assert '<factor id="F2">' in text and '<factor id="F5">' in text
    assert "S1: F2 = a; F5 = a" in text
    assert "C1: F2 (As)" in text and "F1" not in text


def test_factors_are_listed_by_id_not_pool_order():
    text = _render({"facets", "items"}, ids={"A": "F5", "B": "F2"})
    assert text.index('<factor id="F2">') < text.index('<factor id="F5">')
    assert "S1: F2 = a; F5 = a" in text


def test_design_space_guide_follows_the_run_setting_not_the_pool():
    single = _render({"facets", "schemas"}, pool=[], history=[])
    pairs = _render({"facets", "schemas"}, pool=[], history=[], max_schema_facets=2)
    assert "A combination is a single factor." in single
    assert "A combination is a single factor or a pair of factors." in pairs
    assert "kept 0 of 0 single factors and 0 of 0 pairs." in pairs


def test_history_renders_title_definition_and_why_not_kept():
    text = _render({"history"})
    assert "Knife Type" in text and "Which knife is used." in text
    assert "<why_not_kept>no usable values could be built from the tutorials</why_not_kept>" in text
    assert '<why_not_kept>duplicate of "Rice Type"</why_not_kept>' in text


def test_empty_history_renders_placeholder():
    text = _render({"history"}, history=[])
    assert SECTION_HISTORY in text
    assert "None yet." in text


def test_refusal_parsing():
    assert process_facet_iterative_response(
        {"response": {"candidates": [], "refusal_reason": "all duplicates"}}
    ) == []


# ---------------------------------------------------------------------------
# Config: component sets and defaults
# ---------------------------------------------------------------------------

def test_batch_identity_ignores_context():
    a = make_config(strategy=ProposalStrategy.BATCH, context=A1_CONTEXT)
    b = make_config(strategy=ProposalStrategy.BATCH, context=["facets", "items"])
    assert a.semantic_dict() == b.semantic_dict() and "context" not in a.semantic_dict()
    assert "context" in make_config(context=A2_CONTEXT).semantic_dict()


def test_canonical_context_sets():
    assert A1_CONTEXT == frozenset({ContextComponent.FACETS, ContextComponent.ITEMS, ContextComponent.HISTORY})
    assert A2_CONTEXT == frozenset({ContextComponent.FACETS, ContextComponent.SCHEMAS,
                                    ContextComponent.HISTORY})


def test_context_accepts_strings_and_normalizes():
    cfg = make_config(context=["items", "facets", "history", "facets"])
    assert cfg.context == A1_CONTEXT


@pytest.mark.parametrize("bad", [[], ["tree"], ["facets", "nope"], ["schemas", "history"]])
def test_invalid_context_rejected(bad):
    with pytest.raises(ValueError):
        make_config(context=bad)


def test_config_round_trip_and_defaults():
    cfg = make_config(context=A2_CONTEXT, irr_sample_videos=3)
    data = cfg.to_dict()
    assert data["context"] == ["facets", "history", "schemas"]
    assert PipelineConfig.from_dict(data).to_dict() == data

    defaults = PipelineConfig.from_dict({
        k: data[k] for k in ("proposal_model", "vocab_model", "labeling_model",
                             "embedding_model", "item_scope")
    })
    assert defaults.context == A1_CONTEXT
    assert defaults.irr_sample_videos == 3
    assert defaults.dedup_similarity_threshold == 0.93
    assert defaults.seed == 0
    assert defaults.no_improvement_patience == 5
    assert defaults.refusal_patience == 5
    assert defaults.attempt_limit == 3 * defaults.proposal_budget


def test_config_from_dict_rejects_unknown_keys():
    data = make_config().to_dict()
    data["max_iterations"] = 5
    with pytest.raises(ValueError):
        PipelineConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Prompt fingerprint (rendered on a fixed fixture)
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable():
    cfg = make_config()
    assert prompt_fingerprint(cfg) == prompt_fingerprint(make_config())


def test_fingerprint_depends_on_components_and_strategy():
    a1 = prompt_fingerprint(make_config(context=A1_CONTEXT))
    a2 = prompt_fingerprint(make_config(context=A2_CONTEXT))
    a0 = prompt_fingerprint(make_config(strategy=ProposalStrategy.BATCH))
    assert len({a0, a1, a2}) == 3


def test_fingerprint_ignores_settings_outside_prompts():
    assert prompt_fingerprint(make_config(proposal_budget=10, pool_size=10)) == \
        prompt_fingerprint(make_config(proposal_budget=20, pool_size=5))


@pytest.mark.parametrize("module, name", [
    (formatters, "_FACET_FORMAT"),            # formatter logic/template
    (fa_facet_init, "USER_PROMPT"),           # a0 proposal
    (fa_facet_labeling, "LABELING_USER_PROMPT"),
    (fa_facet_vocabulary, "CONSOLIDATION_USER_PROMPT"),
])
def test_fingerprint_changes_when_rendered_prompt_changes(monkeypatch, module, name):
    for strategy in (ProposalStrategy.BATCH, ProposalStrategy.ITERATIVE):
        cfg = make_config(strategy=strategy)
        before = prompt_fingerprint(cfg)
        with monkeypatch.context() as m:
            m.setattr(module, name, getattr(module, name) + "\nEXTRA")
            after = prompt_fingerprint(cfg)
        if module is fa_facet_init and strategy == ProposalStrategy.ITERATIVE:
            assert after == before  # iterative runs never send the init prompt
        else:
            assert after != before, (name, strategy)


def test_fingerprint_changes_when_formatter_code_changes(monkeypatch):
    cfg = make_config()
    before = prompt_fingerprint(cfg)
    original = formatters.format_vocabulary
    monkeypatch.setattr(formatters, "format_vocabulary",
                        lambda *a, **k: original(*a, **k).upper())
    assert prompt_fingerprint(cfg) != before


# ---------------------------------------------------------------------------
# VLM proposer adapter
# ---------------------------------------------------------------------------

def test_vlm_proposer_batch_passes_budget(monkeypatch):
    proposer = VLMFacetProposer(model_config=make_config().proposal_model)
    seen = {}
    monkeypatch.setattr(proposer._extractor, "init_facets",
                        lambda **kw: seen.update(kw) or [make_facet("A")])
    kb = make_kb()
    out = proposer.propose_batch(BatchContext(task="t", videos=kb.by_video(), num_facets=7))
    assert seen["num_facets"] == 7 and [f.id for f in out] == ["A"]


def test_vlm_proposer_iterative_passes_component_names_and_history(monkeypatch):
    proposer = VLMFacetProposer(model_config=make_config().proposal_model)
    seen = {}
    monkeypatch.setattr(proposer._extractor, "propose_facet_iterative",
                        lambda **kw: seen.update(kw) or [])
    history = [HistoryEntry(title="X", scope="global", definition="d", outcome="unreliable")]
    ctx = IterativeContext(task="t", components=A2_CONTEXT, corpus=make_kb().by_video(),
                           knowledge_base=make_kb(), history=history, seed=4)
    assert proposer.propose_one(ctx) == []
    assert seen["components"] == {"facets", "schemas", "history"}
    assert seen["history"] == history and seen["seed"] == 4
    form_facet_iterative_request(**{k: v for k, v in seen.items() if k != "seed"})
