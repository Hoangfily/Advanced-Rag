RAG_ANSWER_TEMPLATE = """Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời dựa trên ngữ cảnh trên. Nếu ngữ cảnh không đủ thông tin, hãy nói rõ điều đó.
"""

QUERY_REWRITE_TEMPLATE = """Viết lại câu hỏi sau cho rõ ràng và cụ thể hơn, giữ nguyên ý định gốc.

Câu hỏi gốc: {question}
Câu hỏi viết lại:"""

MULTI_HOP_DECOMPOSE_TEMPLATE = """Phân tách câu hỏi sau thành các câu hỏi con cần trả lời tuần tự để có đủ thông tin trả lời câu hỏi gốc.
Nếu câu hỏi đã đủ đơn giản, không cần tách, chỉ trả về chính câu hỏi đó.

Câu hỏi: {question}
Danh sách câu hỏi con (mỗi dòng một câu):"""

QUERY_EXPANSION_TEMPLATE = """Viết {num_variants} cách diễn đạt khác nhau cho câu hỏi sau, giữ nguyên ý nghĩa nhưng dùng từ ngữ/góc nhìn khác để tăng khả năng tìm kiếm tài liệu liên quan.

Câu hỏi gốc: {question}
Danh sách cách diễn đạt khác (mỗi dòng một câu, không đánh số):"""
