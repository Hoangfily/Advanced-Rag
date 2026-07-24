import re

from app.generation.prompt_templates import QUERY_EXPANSION_TEMPLATE


def expand_query(query: str, llm_client, num_variants: int = 3) -> list[str]:
    prompt = QUERY_EXPANSION_TEMPLATE.format(question=query, num_variants=num_variants)
    response = llm_client.complete(prompt)

    variants = []
    for line in response.splitlines():
        cleaned = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
        if cleaned.endswith("?"):
            variants.append(cleaned)

    return variants[:num_variants]
