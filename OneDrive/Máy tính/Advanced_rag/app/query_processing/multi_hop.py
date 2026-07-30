from app.generation.llm_client import LLMClient
from app.generation.prompt_templates import MULTI_HOP_DECOMPOSE_TEMPLATE
from app.query_processing.output_parsers import QuestionListOutputParser


def decompose_query(query: str, llm_client: LLMClient) -> list[str]:
    chain = MULTI_HOP_DECOMPOSE_TEMPLATE | llm_client | QuestionListOutputParser()
    sub_queries = chain.invoke({"question": query})

    if not sub_queries:
        return [query]
    return sub_queries
