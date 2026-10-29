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

SECTION_FACETS = "- Existing contextual factors:"
SECTION_ITEMS = "- Tutorial segments (with current pool labels):"
SECTION_SCHEMAS = "- Accepted and rejected facet combinations:"
SECTION_HISTORY = "- Previously tried factors that were not kept:"
ALL_SECTIONS = {
    "facets": SECTION_FACETS,
    "items": SECTION_ITEMS,
    "schemas": SECTION_SCHEMAS,
    "history": SECTION_HISTORY,
}


def _user_text(request) -> str:
    content = request["messages"][1]["content"].content
    return "\n".join(b["text"] for b in content if b["type"] == "text")


def _render(components, *, pool=None, history=None, with_schemas=True):
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
    return _user_text(form_facet_iterative_request(
        task="sushi",
        facets=pool,
        components=components,
        videos=kb.by_video(),
        annotations=ann,
        knowledge_base=kb,
        accepted_schemas=accepted,
        rejected_schemas=rejected,
        history=history if history is not None else [
            HistoryEntry(title="Knife Type", scope="global",
                         definition="Which knife is used.", outcome="vocabulary_failed"),
            HistoryEntry(title="Rice Kind", scope="global",
                         definition="Kind of rice.", outcome="duplicate",
                         duplicate_of="Rice Type"),
        ],
    ))


# ---------------------------------------------------------------------------
# Prompt rendering per component
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("components", [
    {"facets"}, {"items"}, {"schemas"}, {"history"},
    {"facets", "items"}, {"facets", "items", "schemas", "history"},
])
def test_each_component_renders_only_when_requested(components):
    text = _render(components)
    for name, header in ALL_SECTIONS.items():
        assert (header in text) == (name in components), name


def test_substring_context_is_rejected():
    with pytest.raises(TypeError):
        _render("facets+items")


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError):
        _render({"facets", "tree"})


def test_a1_empty_pool_first_iteration():
    text = _render({"facets", "items"}, pool=[], history=[])
    assert "No existing contextual factors." in text
    assert SECTION_ITEMS in text and "Segment i1" in text


def test_a2_empty_pool_first_iteration_has_no_data():
    text = _render({"schemas"}, pool=[], history=[])
    assert "No accepted facet combinations." in text
    assert "No rejected facet combinations." in text
    assert SECTION_ITEMS not in text and "Segment i1" not in text


def test_a2_renders_schema_feedback():
    text = _render({"schemas"})
    assert '<facet_combination id="A1" status="accepted">' in text
    assert "Redundant with accepted facet combination A1" in text
    assert "<signature_frequencies>" in text


def test_history_renders_title_definition_and_outcome():
    text = _render({"history"})
    assert "Knife Type" in text and "Which knife is used." in text
    assert "vocabulary_failed" in text
    assert "duplicate of Rice Type" in text


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

def test_canonical_context_sets():
    assert A1_CONTEXT == frozenset({ContextComponent.FACETS, ContextComponent.ITEMS})
    assert A2_CONTEXT == frozenset({ContextComponent.SCHEMAS})


def test_context_accepts_strings_and_normalizes():
    cfg = make_config(context=["items", "facets"])
    assert cfg.context == A1_CONTEXT


@pytest.mark.parametrize("bad", [[], ["tree"], ["facets", "nope"]])
def test_invalid_context_rejected(bad):
    with pytest.raises(ValueError):
        make_config(context=bad)


def test_config_round_trip_and_defaults():
    cfg = make_config(context=A2_CONTEXT, irr_sample_videos=3)
    data = cfg.to_dict()
    assert data["context"] == ["schemas"]
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
    history = [HistoryEntry(title="X", scope="global", definition="d", outcome="irr_pruned")]
    ctx = IterativeContext(
        task="t", components=frozenset({ContextComponent.SCHEMAS, ContextComponent.HISTORY}),
        knowledge_base=make_kb(), history=history,
    )
    assert proposer.propose_one(ctx) == []
    assert seen["components"] == {"schemas", "history"}
    assert seen["history"] == history
