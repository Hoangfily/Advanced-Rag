from langchain_core.prompts import PromptTemplate

RAG_ANSWER_TEMPLATE = PromptTemplate.from_template(
    """You are an assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer based on the context above. If the context doesn't contain enough information, say so clearly.
Answer in the same language the question was asked in.
"""
)

QUERY_REWRITE_TEMPLATE = PromptTemplate.from_template(
    """Rewrite the following question to be clearer and more specific, while keeping its original intent.
Output ONLY the single rewritten question itself, on one line, with no explanations, no multiple options, \
and no markdown formatting.

Original question: {question}
Rewritten question:"""
)

MULTI_HOP_DECOMPOSE_TEMPLATE = PromptTemplate.from_template(
    """Break down the following question into sub-questions that need to be answered sequentially to gather \
enough information to answer the original question.
If the question is already simple enough, no decomposition is needed — just return the question itself.

Question: {question}
List of sub-questions (one per line):"""
)

QUERY_EXPANSION_TEMPLATE = PromptTemplate.from_template(
    """Write {num_variants} different phrasings of the following question, keeping the same meaning but using \
different wording/perspective to improve the chances of finding relevant documents.

Original question: {question}
List of alternative phrasings (one per line, not numbered):"""
)
