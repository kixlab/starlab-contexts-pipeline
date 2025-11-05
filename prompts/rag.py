from helpers import get_response_pydantic
from pydantic_models.rag import InformationListSchema

SYSTEM_PROMPT_RAG = """You are a helpful assistant that can retrieve information from a library of tutorials for a procedural task. You only consider the provided information (no external knowledge) when answering the query."""

### e.g., query: "Given a tutorial, retrieve all missing, but relevant explanations for the tutorial."
USER_PROMPT_FULL_TUTORIAL = """
Here are the available tutorials for the task `{task}`:
<library>
{library}
</library>

You are given a context tutorial and a query. Please answer the query.
<query>
{query}
</query>

<context tutorial>
{context_tutorial}
</context tutorial>
"""

def tutorials_to_str(tutorials):
    TUTORIAL_FORMAT = "<tutorial idx={idx} title={title}>\n{content}\n</tutorial>"
    text = ""
    for idx, tutorial in enumerate(tutorials):
        text += TUTORIAL_FORMAT.format(idx=idx+1, title=tutorial['title'], content=tutorial['content']) + "\n"
    return text

def get_rag_response_full_tutorial(task, tutorials, tutorial, query, gen_model):
    
    library_str = tutorials_to_str(tutorials)
    context_tutorial_str = tutorial["content"]

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FULL_TUTORIAL.format(task=task, library=library_str, context_tutorial=context_tutorial_str, query=query),
        },
    ]
    response = get_response_pydantic(messages, InformationListSchema, model=gen_model)
    return response["information_list"]

### e.g., query: "Given a tutorial with the highlighted segment, retrieve top-{N} missing, but relevant explanations for the segment"
USER_PROMPT_TUTORIAL_SEGMENT = """
Here are the available tutorials for the task `{task}`:
<library>
{library}
</library>

You are given a context tutorial and its highlighted segment. Please answer the query.

<context tutorial>
{context_tutorial}
</context tutorial>

<highlighted_segment>
{highlighted_segment}
</highlighted_segment>
"""

def get_rag_response_tutorial_segment(task, tutorials, tutorial, segment, query, gen_model):
    
    library_str = tutorials_to_str(tutorials)
    highlighted_segment_str = segment["content"]
    context_tutorial_str = tutorial["content"]

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TUTORIAL_SEGMENT.format(task=task, library=library_str, context_tutorial=context_tutorial_str, highlighted_segment=highlighted_segment_str, query=query),
        },
    ]
    response = get_response_pydantic(messages, InformationListSchema, model=gen_model)
    return response["information_list"]