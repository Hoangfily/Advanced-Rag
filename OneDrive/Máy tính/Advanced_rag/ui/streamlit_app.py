import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Advanced RAG", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --accent: #2b6e64;
        --accent-soft: #e9f1ef;
        --mark: #c98a2c;
        --ink-soft: #5b6a68;
    }
    .block-container { padding-top: 2rem; max-width: 900px; }
    h1 { font-weight: 700; letter-spacing: -0.01em; }
    .app-subtitle { color: var(--ink-soft); font-size: 0.95rem; margin-top: -0.6rem; margin-bottom: 1.6rem; }
    section[data-testid="stSidebar"] .stButton button {
        background: var(--accent); color: white; border: none; width: 100%;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background: #1d4a43; }
    .stChatMessage { border-radius: 14px; }
    .context-card {
        border: 1px solid rgba(43, 110, 100, 0.35);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        background: var(--secondary-background-color, var(--accent-soft));
        color: var(--text-color, #20282b);
    }
    .context-card p { color: var(--text-color, #20282b); }
    .context-source { font-weight: 600; font-size: 0.85rem; color: var(--accent); }
    .score-pill {
        display: inline-block; background: var(--mark); color: white;
        border-radius: 999px; padding: 0.05rem 0.6rem; font-size: 0.75rem;
        font-variant-numeric: tabular-nums; margin-left: 0.5rem;
    }
    .subquery-item { margin: 0.15rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Advanced RAG")
st.markdown(
    '<p class="app-subtitle">Trợ lý hỏi đáp dựa trên tài liệu nội bộ — hybrid retrieval + rerank + Gemini</p>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []


def call_api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Không kết nối được tới API backend. Backend đã chạy chưa? (uvicorn app.main:app)"
    except requests.exceptions.HTTPError as e:
        return None, f"Backend báo lỗi: {e}"
    except Exception as e:
        return None, f"Lỗi không xác định: {e}"


with st.sidebar:
    st.header("📥 Ingest tài liệu")
    st.caption("Mỗi dòng một đường dẫn file, phía server (.txt hoặc .pdf)")
    paths_input = st.text_area("Đường dẫn file", label_visibility="collapsed", height=100)
    if st.button("Ingest", use_container_width=True):
        paths = [p.strip() for p in paths_input.splitlines() if p.strip()]
        if not paths:
            st.warning("Chưa nhập đường dẫn file nào.")
        else:
            with st.spinner("Đang ingest..."):
                result, error = call_api("POST", "/ingest", json={"paths": paths})
            if error:
                st.error(error)
            else:
                st.success(f"Đã ingest **{result['ingested_chunks']}** chunks.")

    st.divider()
    if st.session_state.history and st.button("🗑️ Xoá lịch sử hội thoại", use_container_width=True):
        st.session_state.history = []
        st.rerun()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar="📚"):
        st.write(turn["answer"])

        if turn.get("sub_queries"):
            with st.expander(f"🔎 {len(turn['sub_queries'])} câu hỏi con (multi-hop)"):
                for sub_query in turn["sub_queries"]:
                    st.markdown(f"<div class='subquery-item'>— {sub_query}</div>", unsafe_allow_html=True)

        contexts = turn.get("contexts", [])
        if contexts:
            with st.expander(f"📄 {len(contexts)} đoạn ngữ cảnh đã dùng"):
                for chunk in contexts:
                    st.markdown(
                        f"""<div class="context-card">
                            <span class="context-source">{chunk['source']}</span>
                            <span class="score-pill">score {chunk['score']:.3f}</span>
                            <p style="margin-top:0.5rem; margin-bottom:0;">{chunk['text']}</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Đặt câu hỏi về tài liệu đã ingest...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            data, error = call_api("POST", "/query", json={"question": question})
        if error:
            st.error(error)
        else:
            st.write(data["answer"])
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": data["answer"],
                    "sub_queries": data.get("sub_queries", []),
                    "contexts": data.get("contexts", []),
                }
            )
            st.rerun()
