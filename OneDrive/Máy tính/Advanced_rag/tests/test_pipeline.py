from app.ingestion.chunking import chunk_text


def test_chunk_text_respects_overlap():
    text = "a" * 100
    chunks = chunk_text(text, source="doc.txt", chunk_size=40, chunk_overlap=10)

    assert chunks[0].text == text[0:40]
    assert chunks[1].text == text[30:70]
