import asyncio

from helpers import random_uid
from helpers import llm_enqueue_request
from helpers import llm_get_responses
from helpers import process_api_requests

def batch_run_lm_calls(request_args, func_request, func_response):
    batch_id = random_uid()
    print(f"Batch ID for {func_request.__name__}: {batch_id}")
    if len(request_args) == 0:
        return []
    req_ids = []
    for request_arg in request_args:
        api_args = func_request(**request_arg)
        req_id = llm_enqueue_request(batch_id, **api_args)
        req_ids.append(req_id)

    asyncio.run(process_api_requests(batch_id))
    batch_responses = llm_get_responses(batch_id)

    batch_results = []
    for request_arg, req_id in zip(request_args, req_ids):
        if req_id not in batch_responses:
            raise ValueError(f"Request ID not found: {req_id}")
        response = batch_responses[req_id]
        result = func_response(response=response, **request_arg)
        batch_results.append(result)
    return batch_results