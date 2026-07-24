import re

from app.generation.prompt_templates import MULTI_HOP_DECOMPOSE_TEMPLATE


def decompose_query(query: str, llm_client) -> list[str]:
    prompt = MULTI_HOP_DECOMPOSE_TEMPLATE.format(question=query)
    response = llm_client.complete(prompt)

    sub_queries = []
    for line in response.splitlines():
        cleaned = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
        if cleaned.endswith("?"):
            sub_queries.append(cleaned)

    if not sub_queries:
        return [query]
    return sub_queries
