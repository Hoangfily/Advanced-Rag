from app.generation.prompt_templates import QUERY_REWRITE_TEMPLATE


def rewrite_query(query: str, llm_client) -> str:
    prompt = QUERY_REWRITE_TEMPLATE.format(question=query)
    response = llm_client.complete(prompt)
    return response.strip().strip('"')
