from app.generation.llm_client import LLMClient
from app.generation.prompt_templates import QUERY_REWRITE_TEMPLATE
from app.query_processing.output_parsers import SingleQuestionOutputParser


def rewrite_query(query: str, llm_client: LLMClient) -> str:
    chain = QUERY_REWRITE_TEMPLATE | llm_client | SingleQuestionOutputParser()
    return chain.invoke({"question": query})
