"""Shared factor criteria, factor kinds, and kind-aware labeling."""

from __future__ import annotations

from facets import Facet, FacetLabel
from facets.prompts import fa_facet_init, fa_facet_iterative, fa_facet_labeling
from facets.prompts.fa_criteria import FACTOR_CRITERIA
from facets.prompts.fa_facet_init import process_facet_init_response
from facets.prompts.fa_facet_iterative import process_facet_iterative_response
from facets.prompts.formatters import format_facet


def _candidate(kind="moment"):
    return {"title": "Roll Sealing", "title_plural": "Roll Sealings",
            "user_question": "How are you sealing the roll?", "scope": "fine-grained",
            "kind": kind, "definition": "d", "justification": "j", "guidelines": ["g"],
            "vocabulary": [{"label": "Water", "definition": "w"},
                           {"label": "not applicable", "definition": "n"}]}


def test_all_proposal_prompts_share_the_criteria_and_drop_the_depth_rule():
    for text in (fa_facet_init.USER_PROMPT, fa_facet_iterative._INSTRUCTIONS):
        assert FACTOR_CRITERIA in text
        assert "DEPTH RULE" not in text and "finer-grained" not in text


def test_criteria_define_three_kinds_and_their_special_values():
    for kind in ("situation:", "progress:", "moment:"):
        assert kind in FACTOR_CRITERIA
    assert '"unspecified"' in FACTOR_CRITERIA and '"not applicable"' in FACTOR_CRITERIA


def test_kind_is_parsed_from_both_proposal_responses():
    response = {"response": {"candidates": [_candidate("situation")], "refusal_reason": ""}}
    assert process_facet_init_response(response)[0].kind == "situation"
    assert process_facet_iterative_response(response)[0].kind == "situation"


def test_kind_round_trips_and_is_shown_to_the_model():
    f = Facet(id="f", title="T", title_plural="Ts", scope="global", definition="d",
              justification="j", guidelines=[], vocabulary=[FacetLabel("a", "a")], kind="progress")
    assert Facet.from_dict(f.to_dict()).kind == "progress"
    assert "<kind>progress</kind>" in format_facet(f, "F1")
    legacy = Facet.from_dict({k: v for k, v in f.to_dict().items() if k != "kind"})
    assert legacy.kind == "" and "<kind>" not in format_facet(legacy, "F1")


def test_labeling_prompt_explains_each_kind():
    text = fa_facet_labeling.LABELING_USER_PROMPT
    assert "situation:" in text and "progress:" in text and "moment:" in text
    assert "closest match" not in text
