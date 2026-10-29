"""
Every message a proposer can receive, rendered on dummy data, for review.

Builds a small sushi-roll corpus and plays a scripted sequence of ten
submissions that covers every candidate outcome (accepted, duplicate,
unreliable, replaced, schema-rejected, vocabulary and labeling failures). Pool
decisions are scripted, but the design space, history, factor ids and every
message are produced by the real runner code (``PipelineRunner`` and
``AgenticRunner`` with no model behind them). Writes one Markdown file:

    python -m reports.prompt_catalog [--out docs/prompt_catalog.md]
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from contextual import bookkeeping
from contextual.evaluation import PoolEvaluator
from contextual.models import CandidateStatus, RunState
from facets import Facet, FacetLabel
from facets.prompts.fa_facet_init import form_facet_init_request
from facets.prompts.fa_facet_iterative import form_facet_iterative_request
from knowledge_items import ItemScopeConfig, KnowledgeItemType
from knowledge_items.schema import KnowledgeBase, KnowledgeItem, KnowledgeSource
from utils import ModelConfig
from utils.llm import MessageContent
from utils.tool_loop import ToolCall

from agentic.config import AgenticConfig, FeedbackLevel
from agentic.feedback import render_feedback
from agentic.prompts import ONE_AT_A_TIME, kickoff, nudge
from agentic.runner import AgenticRunner
from agentic.tools import READ_LABELS
from pipeline.config import A1_CONTEXT, A2_CONTEXT, PipelineConfig, ProposalStrategy
from pipeline.runner import PipelineRunner

TASK = "making a sushi roll"
BUDGET = 10
POOL_SIZE = 4

VIDEOS = {
    "maki_basics": ["Rinse the rice until the water runs clear.",
                    "Spread rice on the nori, leaving an inch bare at the top.",
                    "Lay cucumber and salmon in a line across the rice.",
                    "Roll tightly with the bamboo mat and seal the edge with water."],
    "inside_out": ["Season the cooked rice with sushi vinegar while it is warm.",
                   "Cover the nori with rice, then flip it so the rice faces down.",
                   "Add crab and avocado along the middle.",
                   "Roll with the mat and coat the outside with sesame seeds."],
    "hand_roll": ["Cook the rice in a rice cooker and let it cool.",
                  "Hold a half sheet of nori in your palm.",
                  "Add tuna and a spoonful of rice at one corner.",
                  "Fold into a cone and seal it with a grain of rice."],
}

# title -> (kind, question, values, labels per segment in corpus order)
FACTORS = {
    "Roll Type": ("situation", "What kind of roll are you making?",
                  ["maki", "inside-out", "hand roll"],
                  ["maki"] * 4 + ["inside-out"] * 4 + ["hand roll"] * 4),
    "Step": ("progress", "Which step are you on?",
             ["preparing rice", "spreading rice", "adding filling", "rolling"],
             ["preparing rice", "spreading rice", "adding filling", "rolling"] * 3),
    "Filling Protein": ("situation", "Which protein goes in your roll?",
                        ["salmon", "crab", "tuna"],
                        ["salmon"] * 4 + ["crab"] * 4 + ["tuna"] * 4),
    "Sealing Method": ("moment", "How are you sealing the roll?",
                       ["water", "rice grain", "not applicable"],
                       ["not applicable"] * 3 + ["water"] + ["not applicable"] * 7
                       + ["rice grain"]),
    "Rice Seasoning": ("moment", "Is your rice seasoned yet?",
                       ["plain", "seasoned", "not applicable"],
                       ["plain"] + ["not applicable"] * 3 + ["seasoned"]
                       + ["not applicable"] * 3 + ["plain"] + ["not applicable"] * 3),
    "Roll Style": ("situation", "Which style of roll is it?", ["thin", "thick"], None),
    "Hand Comfort": ("situation", "How comfortable are your hands with rolling?",
                     ["nervous", "confident"], None),
    "Mat Use": ("moment", "Are you using a bamboo mat?",
                ["mat", "no mat", "not applicable"], None),
    "Garnish": ("moment", "Which garnish are you adding?", ["sesame", "none"], None),
    "Nori Size": ("situation", "What size of nori sheet do you have?", ["full", "half"], None),
}

# (title, status, replaced title); played in order
SCRIPT: List[Tuple[str, CandidateStatus, Optional[str]]] = [
    ("Roll Type", CandidateStatus.ACCEPTED, None),
    ("Step", CandidateStatus.ACCEPTED, None),
    ("Roll Style", CandidateStatus.DUPLICATE, None),
    ("Filling Protein", CandidateStatus.ACCEPTED, None),
    ("Hand Comfort", CandidateStatus.IRR_PRUNED, None),
    ("Sealing Method", CandidateStatus.ACCEPTED, None),
    ("Rice Seasoning", CandidateStatus.REPLACED, "Filling Protein"),
    ("Mat Use", CandidateStatus.SCHEMA_REJECTED, None),
    ("Garnish", CandidateStatus.VOCABULARY_FAILED, None),
    ("Nori Size", CandidateStatus.LABELING_FAILED, None),
]


def _kb() -> KnowledgeBase:
    items, n = [], 0
    for video, texts in VIDEOS.items():
        for text in texts:
            n += 1
            items.append(KnowledgeItem(id=f"seg{n}", type=KnowledgeItemType.STEP, text=text,
                                       source=KnowledgeSource(video_id=video), unit_id=f"u{n}"))
    return KnowledgeBase(items=items)


def _facet(title: str) -> Facet:
    kind, question, values, _ = FACTORS[title]
    return Facet(
        id=title.lower().replace(" ", "_"), title=title, title_plural=f"{title}s",
        user_question=question, scope="global", kind=kind,
        definition=f"The {title.lower()} of the learner's attempt.",
        justification="Dummy factor for the prompt catalog.",
        guidelines=[f"Label the {title.lower()} the tutorial shows or implies."],
        vocabulary=[FacetLabel(label=v, definition=f"The learner's {title.lower()} is {v}.")
                    for v in values],
    )


def _settings() -> Dict:
    model = ModelConfig(category="vlm", provider="gemini", model="dummy")
    return dict(proposal_model=model, vocab_model=model, labeling_model=model,
                embedding_model=ModelConfig(category="embedding", provider="gemini",
                                            model="dummy"),
                item_scope=ItemScopeConfig(item_type=KnowledgeItemType.STEP),
                proposal_budget=BUDGET, pool_size=POOL_SIZE, max_schema_facets=2,
                refinement_threshold=0.55, use_cache=False)


def play(kb: KnowledgeBase, evaluator: PoolEvaluator) -> List[RunState]:
    """The state after each scripted submission (states[k] = after k+1)."""
    state, states = RunState(), []
    for title, status, replaced in SCRIPT:
        facet = _facet(title)
        labels = FACTORS[title][3]
        if status in (CandidateStatus.ACCEPTED, CandidateStatus.REPLACED,
                      CandidateStatus.SCHEMA_REJECTED):
            for item, label in zip(kb, labels or ["not applicable"] * len(kb.items)):
                state.annotations[(item.id, facet.id)] = label
        kwargs = {}
        if status == CandidateStatus.ACCEPTED:
            state.pool = state.pool + [facet]
        elif status == CandidateStatus.REPLACED:
            old = _facet(replaced).id
            state.pool = [facet if f.id == old else f for f in state.pool]
            kwargs["replaced_facet_id"] = old
        elif status == CandidateStatus.DUPLICATE:
            kwargs["duplicate_of"] = _facet("Roll Type").id
        state.annotations = state.annotations.filter_facets(state.pool_ids())
        evaluator.initialize(state)
        bookkeeping.record(state, status, facet=facet, **kwargs)
        states.append(copy.deepcopy(state))
    return states


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _text(content) -> str:
    if isinstance(content, MessageContent):
        content = content.content
    if isinstance(content, list):
        return "".join(b["text"] for b in content if b.get("type") == "text")
    return content


def _block(title: str, text: str, level: int = 4) -> List[str]:
    return [f"{'#' * level} {title}", "", "````text", text.rstrip(), "````", ""]


def _request(title: str, request, shared_system: str) -> List[str]:
    system, user = (_text(m["content"]) for m in request["messages"])
    assert system == shared_system, "pipeline system messages must be identical"
    return [f"### {title}", "", "System message: the shared one above.", ""] + \
        _block("User message", user)


def build() -> str:
    kb = _kb()
    tmp = tempfile.mkdtemp()
    pipe = {name: PipelineRunner(
        config=PipelineConfig(**_settings(), strategy=strategy, context=context),
        task=TASK, knowledge_base=kb, proposer=None, vocab_builder=None, labeler=None,
        embedder=None, runs_dir=tmp)
        for name, strategy, context in (("a0", ProposalStrategy.BATCH, A1_CONTEXT),
                                        ("a1", ProposalStrategy.ITERATIVE, A1_CONTEXT),
                                        ("a2", ProposalStrategy.ITERATIVE, A2_CONTEXT))}
    agents = {level: AgenticRunner(
        config=AgenticConfig(**_settings(), feedback=level), task=TASK, knowledge_base=kb,
        client=None, vocab_builder=None, labeler=None, embedder=None, runs_dir=tmp)
        for level in FeedbackLevel}
    states = play(kb, pipe["a1"].evaluator)
    final = states[-1]

    out = ["# Prompt catalog", "",
           "Generated by `python -m reports.prompt_catalog` on dummy data: a 12-segment "
           f"corpus about {TASK}, budget {BUDGET}, pool size {POOL_SIZE}, pairs, "
           "τ = 0.55. Ten scripted submissions cover every outcome:", ""]
    out += [f"{k}. {t}: {s.value}" + (f" (replaces {r})" if r else "")
            for k, (t, s, r) in enumerate(SCRIPT, start=1)]
    out += ["", "| | Pushed | Pulled |", "|---|---|---|",
            "| a0 | — | — |", "| a1 | FACETS + ITEMS + HISTORY (full) | — |",
            "| a2 | FACETS + SCHEMAS + HISTORY (full) | — |",
            "| g0 | receipt | — |", "| g1 | FACETS + HISTORY (what changed) | ITEMS (`read_labels`) |",
            "| g2 | FACETS + HISTORY (what changed) + SCHEMAS (full) | ITEMS (`read_labels`) |",
            ""]

    # ---- pipeline ----
    shared = _text(form_facet_init_request(TASK, pipe["a0"].corpus, BUDGET)["messages"][0]["content"])
    out += ["## Pipeline (a0, a1, a2)", "",
            "Every pipeline request has the same system message: the role and every "
            "segment of the corpus.", ""]
    out += _block("Shared system message", shared, level=3)
    out += _request("a0: the batch request", form_facet_init_request(
        TASK, pipe["a0"].corpus, BUDGET), shared)
    for name in ("a1", "a2"):
        runner = pipe[name]
        out += _request(f"{name}: first request (empty pool)", form_facet_iterative_request(
            **runner._build_iterative_context(RunState(), 1).request_args()), shared)
        out += _request(f"{name}: request after all ten submissions", form_facet_iterative_request(
            **runner._build_iterative_context(final, 11).request_args()), shared)

    # ---- agent ----
    out += ["## Agent (g0, g1, g2)", ""]
    for level, runner in agents.items():
        name = {"none": "g0", "pool": "g1", "schemas": "g2"}[level.value]
        out += [f"### {name} (`feedback: {level.value}`)", ""]
        out += _block("System prompt", runner.system)
        out += _block("Tools", json.dumps([t.to_dict() for t in runner.tools], indent=2))
        out += _block("Kickoff (first user message)", kickoff(TASK))
        out += _block("Nudge (after an early plain reply)", nudge(3))
        if level != FeedbackLevel.NONE:
            for title, state, args in (
                    ("`read_labels` before anything is accepted", RunState(), {}),
                    ("`read_labels` after all ten submissions, one video",
                     final, {"video_ids": ["inside_out"]}),
                    ("`read_labels` with an unknown video", final, {"video_ids": ["nope"]})):
                out += _block(title, runner._handle(state, ToolCall("c", READ_LABELS, args)))
        else:
            out += _block("Calling `read_labels` (not available)",
                          runner._handle(RunState(), ToolCall("c", READ_LABELS, {})))
        for k, ((title, status, replaced), state) in enumerate(zip(SCRIPT, states), start=1):
            label = f"Submission {k}: {title} → {status.value}" + (
                f" (replaces {replaced})" if replaced else "")
            out += _block(label, render_feedback(state.ledger[-1], state, runner.config,
                                                 runner.focused_kb))
        out += _block("A second submission in the same turn (not processed)", ONE_AT_A_TIME)
        invalid = runner._submit(RunState(), {"title": "Knife"})
        out += _block("Invalid submission (missing fields)", invalid)
        spent = RunState(proposals=BUDGET)
        out += _block("Submission after the budget is spent", runner._submit(spent, {}))
        if level != FeedbackLevel.NONE:
            busy = RunState(runner_state={"exploration_since_submit": 20})
            out += _block("Exploration limit reached",
                          runner._handle(busy, ToolCall("c", READ_LABELS, {})))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default="docs/prompt_catalog.md")
    args = parser.parse_args()
    Path(args.out).write_text(build())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
