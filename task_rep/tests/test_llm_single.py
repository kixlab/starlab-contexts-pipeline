"""
Quick tester: send a single LLM request from a logged JSON line.

Usage:
    python -m tests.test_llm_single <path_to_jsonl> [--line N] [--request-id ID] [--model MODEL]

Examples:
    # By line number (1-indexed)
    python -m tests.test_llm_single static/lm-batches/facet_labeling_xxx_llm_responses.jsonl --line 296

    # By request ID
    python -m tests.test_llm_single static/lm-batches/facet_labeling_xxx_llm_responses.jsonl \
        --request-id 7cb6a034-af13-41c0-a7db-306a188ebe51

    # Paste JSON directly via stdin
    echo '{"request_id": "...", "messages": [...], ...}' | python -m tests.test_llm_single -
"""

import argparse
import json
import sys

from utils.llm import get_response_pydantic_openai, MessageContent


def load_request(source: str, line: int | None, request_id: str | None) -> dict:
    """Load a single request JSON from a file or stdin."""
    if source == "-":
        return json.loads(sys.stdin.read())

    with open(source) as f:
        if request_id:
            for raw in f:
                rec = json.loads(raw)
                if rec.get("request_id") == request_id:
                    return rec
            raise ValueError(f"request_id {request_id!r} not found in {source}")

        if line is not None:
            for i, raw in enumerate(f, 1):
                if i == line:
                    return json.loads(raw)
            raise ValueError(f"Line {line} not found (file has {i} lines)")

        # Default: last line
        last = None
        for raw in f:
            last = raw
        return json.loads(last)


def convert_messages(messages: list[dict]) -> list[dict]:
    """Convert serialized MessageContent dicts to OpenAI API format."""
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, dict) and "content" in content and isinstance(content["content"], list):
            content = MessageContent.from_dict(content).to_openai_content()
        out.append({"role": msg["role"], "content": content})
    return out


def main():
    parser = argparse.ArgumentParser(description="Replay a single logged LLM request")
    parser.add_argument("source", help="Path to JSONL file, or '-' for stdin")
    parser.add_argument("--line", type=int, default=None, help="Line number (1-indexed)")
    parser.add_argument("--request-id", default=None, help="Request ID to find")
    parser.add_argument("--model", default=None, help="Override model name")
    args = parser.parse_args()

    record = load_request(args.source, args.line, args.request_id)
    print(f"Request ID: {record['request_id']}")

    messages = convert_messages(record["messages"])
    response_format = record.get("response_format")

    print(f"Messages: {len(messages)} ({', '.join(m['role'] for m in messages)})")
    if response_format:
        print(f"Response format: {response_format.get('name', 'unnamed')}")
    print("---")

    result = get_response_pydantic_openai(
        messages=messages,
        response_format=response_format,
        model=args.model,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
