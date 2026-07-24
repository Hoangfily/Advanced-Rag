from app.generation.prompt_templates import RAG_ANSWER_TEMPLATE


def test_rag_answer_template_formats_context_and_question():
    prompt = RAG_ANSWER_TEMPLATE.format(context="ngữ cảnh X", question="câu hỏi Y")

    assert "ngữ cảnh X" in prompt
    assert "câu hỏi Y" in prompt
