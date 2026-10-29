"""
Prompts for revising existing contextual factors (facets).

Revision can include:
- Adding new values to vocabularies
- Refining definitions and guidelines
- Splitting overly broad facets
- Merging redundant facets
"""
import logging
from typing import Any, Dict, List

from utils.llm import MessageContent
from utils import random_uid

from .fa_pydantic import ContextualFactorRevisionResult
from .formatters import facets_to_message_content

from facets.schema import Facet, FacetLabel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You design user-facing metadata for tutorial videos about: {task}.

You refine contextual factors: dimensions of a learner's situation that determine which tutorial segments apply to them. A learner's situation consists of their goal, materials, equipment, and progress through the task.

Your audience is novice and intermediate learners of {task}. Factors become filter menus, timeline labels, and search options. Learners use them to find useful segments before watching, so every factor must be something a learner can state about their own situation without seeing the footage."""

USER_PROMPT = """
You are given a list of contextual factors that enumerate possible task contexts (e.g., constraints, resources, approaches, goals, etc.) for a particular task. Your goal is to analyze the factors and propose new factors that help reduce the size of the list (i.e., total number of labels across all factors) either by "merging" a set of factors into one or "splitting" a factor into a set of smaller factors. Follow the procedure below.

CONTEXTUAL FACTOR DEFINITION
A valid contextual factor is a variable describing the procedural situation at a given moment or span. It must be:

- Domain-grounded: anchored in the task-world (e.g., procedural stage, material state, tool state, object configuration, environmental condition, operating mode, verification state, service context) — NOT in media, narration, editing, presentation, or social framing.
- Consequential: changes what actions, constraints, interpretations, checks, or information apply.
- Monolithic: captures one coherent semantic concept, not a bundle.
- Piecewise-stable: remains constant over meaningful spans; changes at identifiable boundaries.
- Target-independent: determined by the broader procedural context; removing the target item would not change the factor's value.

Non-examples: video genre, tutorial format, presenter configuration, narration style, camera/editing style, entertainment framing, human skill level, subjective quality, labels restating target content.

WHAT QUALIFIES AS A CONTEXTUAL FACTOR
Each vocabulary value must be a situation a learner could truthfully claim before watching. Test every value with this sentence:

    "Show me the segments for when ___."

- If a value can only be determined by observing how the instructor moves, holds, paces, presents, or what the camera shows, it describes the footage, not a learner's situation.
- A factor is useful only if learners in different situations need different segments. Before proposing it, identify two concrete learners whose situations differ on this factor and who would retrieve different segments.
- Values must use the words of {task}, not vocabulary from robotics, motor control, cinematography, or another discipline.
- If the factor and vocabulary would transfer unchanged to an unrelated manual task such as pottery, knitting, surgery, or furniture assembly, reject it.

EXPRESSION REQUIREMENTS
- `title`: 1-3 everyday words; a concrete noun phrase a learner would recognize.
- `user_question`: the learner-facing question this factor answers, e.g. "What kind of roll are you making?" or "What tool do you have?"
- `vocabulary`: 3-8 values when possible. Labels are 1-3 common words and distinguishable by a first-time learner. One "other" or "not applicable" value is allowed.
- `definition` and value definitions: one plain sentence, roughly 15 words or fewer.
- `guidelines`: annotator-facing edge cases, tie-breakers, boundary conditions, and label-change triggers.

STOPPING RULE
If no merge/split/revision produces a cleaner learner-situation factor, return an empty list.

GOAL
Propose contextual factors to compress the given set of factors. 
Follow the requirements and the procedure below.

WHEN NOT TO PROPOSE FACTORS
Do NOT force factors that are artificial or unreliable. Return an empty candidate list with a refusal reason if any of the following apply:
- All target items occur in the same narrow region of the tutorial and share identical context — no factor can assign them different labels without reading the item content itself (violating target-independence).
- The only distinguishing factors would be too fine-grained to generalize (e.g., "the exact product being discussed"), essentially re-encoding the item content as a label.
- Proposed factors would require subjective or ambiguous judgment calls that different annotators would disagree on.
It is better to return zero candidates than to propose a factor that does not reliably separate the targets.

PROCEDURE
1) Identify opportunities for list compression:
    - Check for a set of factors that could be merged into one monolithic factor.
    - Check for a an factor that could benefit from being split into a set of smaller, monolithic factors that contribute fewer labels in total.
2) For each opportunity identified, propose a new factor that captures the merged/split factors. When defining the factor, do not refer to the merged/split factors, but formulate its own definition that describes what factor of the task context it captures.
    - If there no opportunities identified, return an empty list of candidates.
    - For the "merging" opportunity, propose a new monolithic factor. Justify why merging is effective.
    - For the "splitting" opportunity, propose a set of new, monolithic factors. Justify why splitting is effective.
    - Justify why the proposed factor satisfies each part of the definition of a contextual factor.

OUTPUT
For each revised contextual factor, provide:
- `title`: short name (1-3 words)
- `title_plural`: plural form of the title
- `user_question`: a learner-facing question used to choose the factor value
- `scope`: one of "global", "mid-level", or "fine-grained"
- `definition`: brief description of what this factor captures
- `justification`: (1) why this is a valid contextual factor, and (2) why it is particularly proposed as a revision
- `guidelines`: list of concise instructions for labeling tutorial segments under this factor, including when the label should change
- `vocabulary`: finite list of mutually exclusive label values, each with `label` and `definition`

Also provide a top-level `changes` field: a summary of what was merged, split, or otherwise changed and why. If no candidates are proposed, use `changes` to briefly explain why.

List of existing factors:
{existing_facets}

REMINDER: every value must complete "Show me the segments for when ___"; use learner-situation vocabulary; return no candidates if no cleaner learner-situation factor exists.
"""


def form_facet_revision_request(
    task: str,
    facets: List[Facet],
    **kwargs,
) -> Dict[str, Any]:
    """Form request to revise existing facets."""
    messages = [
        {
            "role": "system",
            "content": MessageContent.from_template(SYSTEM_PROMPT, task=task),
        },
        {
            "role": "user",
            "content": MessageContent.from_template(
                USER_PROMPT,
                task=task,
                existing_facets=facets_to_message_content(facets),
            ),
        },
    ]
    return {
        "messages": messages,
        "response_format": ContextualFactorRevisionResult,
    }


def process_facet_revision_response(
    response: Dict[str, Any],
    **kwargs,
) -> List[Facet]:
    """Process facet revision response, returning typed Facet objects.

    IDs are self-generated (not taken from the LLM response).
    """
    result = response.get("response", {})
    candidates = result.get("candidates", [])
    changes = result.get("changes", "")
    logger.info("Revision changes: %s", changes)
    facets = []
    for candidate in candidates:
        vocabulary = [
            FacetLabel(label=ld["label"].strip().lower(), definition=ld["definition"])
            for ld in candidate.get("vocabulary", [])
        ]
        facets.append(Facet(
            id=random_uid(),
            title=candidate["title"],
            title_plural=candidate["title_plural"],
            user_question=candidate.get("user_question", ""),
            scope=candidate["scope"],
            definition=candidate["definition"],
            justification=candidate["justification"],
            guidelines=candidate["guidelines"],
            vocabulary=vocabulary,
        ))
    return facets
