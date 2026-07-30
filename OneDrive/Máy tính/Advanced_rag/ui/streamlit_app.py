import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Advanced RAG", page_icon="📚")
st.title("Advanced RAG Demo")

with st.sidebar:
    st.header("Ingest tài liệu")
    paths_input = st.text_area("Đường dẫn file (mỗi dòng một file)")
    if st.button("Ingest"):
        paths = [p.strip() for p in paths_input.splitlines() if p.strip()]
        response = requests.post(f"{API_URL}/ingest", json={"paths": paths})
        result = response.json()
        st.markdown(f"Đã ingest **{result['ingested_chunks']}** chunks.")

question = st.text_input("Đặt câu hỏi")
if st.button("Hỏi") and question:
    response = requests.post(f"{API_URL}/query", json={"question": question})
    data = response.json()
    st.subheader("Trả lời")
    st.write(data["answer"])

    if data.get("sub_queries"):
        st.subheader("Câu hỏi con (multi-hop)")
        for sub_query in data["sub_queries"]:
            st.markdown(f"- {sub_query}")

    st.subheader("Ngữ cảnh sử dụng")
    for chunk in data.get("contexts", []):
        with st.expander(f"{chunk['source']} (score={chunk['score']:.4f})"):
            st.markdown(chunk["text"])
