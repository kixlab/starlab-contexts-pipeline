RELEVANCE_CRITERIA_LIKERT_3 = """
- 3: Highly relevant to the query.
- 2: Relevant to the query.
- 1: Not relevant to the query.

Give a score between 1 and 3.
"""

RELEVANCE_CRITERIA_LIKERT_5 = """
- 5: Extremely relevant to the query.
- 4: Highly relevant to the query.
- 3: Relevant to the query.
- 2: Somewhat relevant to the query.
- 1: Not relevant to the query.

Give a score between 1 and 5.
"""

RELEVANCE_CRITERIA_BINARY = """
- yes: The information is relevant to the query.
- no: The information is not relevant to the query.

Give a binary decision between yes and no.
"""

RELEVANCE_CRITERIA_COMPARISON = """
- A: The response from method A is more relevant to the query than the response from method B.
- B: The response from method B is more relevant to the query than the response from method A.
- tie: The response from method A and method B are equally relevant to the query.

Give a decision between A, B, or tie.
"""