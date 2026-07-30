from pydantic import BaseModel, Field

from app.generation.llm_client import LLMClient
from app.generation.prompt_templates import QUERY_EXPANSION_TEMPLATE
from app.query_processing.output_parsers import QuestionListOutputParser


class QueryExpansionConfig(BaseModel):
    num_variants: int = Field(gt=0)


def expand_query(query: str, llm_client: LLMClient, num_variants: int = 3) -> list[str]:
    config = QueryExpansionConfig(num_variants=num_variants)

    chain = QUERY_EXPANSION_TEMPLATE | llm_client | QuestionListOutputParser()
    variants = chain.invoke({"question": query, "num_variants": config.num_variants})

    return variants[:config.num_variants]
