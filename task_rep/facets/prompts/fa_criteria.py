"""
Shared definition of a contextual factor, used by every proposal prompt (batch,
iterative, and the agentic system prompt) so the criteria cannot drift apart.

Format with ``task``.
"""

FACTOR_CRITERIA = """WHAT A CONTEXTUAL FACTOR IS
A contextual factor is a question learners can answer about their own situation to find the tutorial segments that fit them. Every value must complete the sentence "Show me the segments for when ___." There are three kinds, and all are useful:
- situation: holds for the learner's whole attempt: what they are making, the ingredients, materials, and equipment they have, and their constraints. Example: "What kind of roll are you making?" Every segment of a tutorial has a value.
- progress: where the learner is in the task. Example: "Which step are you on?" Every segment has a value. One progress factor is usually enough.
- moment: a choice or problem the learner faces during one part of the task. Example: "How are you sealing the roll?" Only segments of that part have a value; all others are "not applicable".

QUALITY
- A factor is useful only if learners who answer it differently need different segments. Name two such learners.
- Use the words of {task} and everyday language a first-time learner knows, not vocabulary from robotics, motor control, cinematography, or another discipline. If the factor would transfer unchanged to an unrelated task such as pottery, knitting, surgery, or furniture assembly, it is too generic.
- Never describe the footage: how the instructor moves or holds things, pacing, camera, narration, editing, or what is visible.
- Distinct from the other factors and not predictable from their combination.
- Grounded in the segments, with at least two values that actually occur.
- Prefer broad coverage. A situation or progress factor should have a clear value in most tutorials; a moment factor should concern a part of the task that most tutorials include. Go narrower only when nothing broader remains.

EXPRESSION
- `title`: 1-3 everyday words; a noun phrase a learner recognizes.
- `user_question`: the question the learner answers, addressed to them ("What ... are you ...?").
- `kind`: situation, progress, or moment, as defined above.
- `scope`: global, mid-level, or fine-grained.
- `vocabulary`: 3-8 mutually exclusive values of 1-3 common words. A situation factor may add "unspecified" for tutorials that never reveal it; a moment factor must add "not applicable"; a progress factor has neither.
- `definition` and value definitions: one plain sentence each, about 15 words or fewer.
- `guidelines`: edge cases and tie-breakers for the people labeling segments.
- `justification`: the two learners, and why the factor is distinct from the existing ones.

EXAMPLES (different domain: pottery)
- BAD: "Prehension Patterns" / "How is the object grasped?" / precision pinch, power grasp. This describes the footage.
- GOOD situation: "Clay Body" / "What clay are you using?" / stoneware, porcelain, earthenware, unspecified.
- GOOD progress: "Clay Stage" / "How firm is your clay right now?" / wet, leather-hard, bone-dry, fired.
- GOOD moment: "Handle Joining" / "How are you attaching the handle?" / slip and score, pulled on, not applicable."""
