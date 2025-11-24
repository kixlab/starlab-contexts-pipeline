RELEVANCE_CRITERIA_LIKERT_3 = """
- 3: Highly relevant and helpful information — crucial for learning and completing the task.
- 2: Relevant, but not helpful — contributes somewhat to learning and completing the task but is not essential.
- 1: Not relevant or already present in the current tutorial — not useful for learning and completing the task.

Give a score between 1 and 3.
"""

RELEVANCE_CRITERIA_LIKERT_5 = """
- 5: Extremely relevant and highly helpful information — crucial for learning and completing the task.
- 4: Highly relevant and helpful information — significantly supports learning and completing the task.
- 3: Relevant but moderately helpful information — contributes somewhat to learning and completing the task but is not essential.
- 2: Marginally relevant — information is related but not useful for learning and completing the task.
- 1: Not relevant or already present in the current tutorial — not useful for learning and completing the task.

Give a score between 1 and 5.
"""

RELEVANCE_CRITERIA_BINARY = """
- yes: The information is relevant to the query, useful for learning and completing the task, and is missing from the current tutorial.
- no: The information is not relevant to the query, not useful for learning and completing the task, or is present in the current tutorial.

Give a binary decision between yes and no.
"""

RELEVANCE_CRITERIA_COMPARISON = """
"""

COMPREHENSIVENESS_CRITERIA_COMPARISON = """
"""