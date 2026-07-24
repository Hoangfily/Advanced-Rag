# Advanced RAG

Đồ án cuối kì: hệ thống RAG nâng cao, dùng Gemini API (Google) làm embedding + LLM, ChromaDB làm vector store.

Kỹ thuật đã triển khai:
- **Hybrid search**: kết hợp dense retrieval (embedding, Chroma) và sparse retrieval (BM25), hợp nhất bằng Reciprocal Rank Fusion.
- **Re-ranking**: cross-encoder đa ngôn ngữ (`sentence-transformers`) chấm điểm lại và dedupe kết quả trước khi đưa vào LLM.
- **Query rewriting**: viết lại câu hỏi gốc cho rõ ràng hơn trước khi truy xuất.
- **Multi-hop reasoning**: phân tách câu hỏi phức tạp thành các câu hỏi con, truy xuất riêng từng câu.
- **Query expansion**: sinh thêm các cách diễn đạt khác của câu hỏi để tăng độ phủ truy xuất (tắt mặc định, xem phần Cấu hình).

## Cấu trúc thư mục

```
app/
  main.py                    # FastAPI entrypoint
  api/routes/                 # health, ingest, query endpoints
  core/                       # config (Settings đọc từ .env), logging
  ingestion/                  # load file (.txt/.pdf), chunking, Gemini embedder
  retrieval/                  # Chroma vector store, BM25 sparse, hybrid fusion (RRF), cross-encoder reranker
  query_processing/           # query rewriting, expansion, multi-hop decomposition (gọi Gemini)
  generation/                 # prompt templates, Gemini LLM client
  pipeline/                   # RagPipeline điều phối toàn bộ luồng ingest + query
  schemas/                    # Pydantic request/response models
ui/
  streamlit_app.py            # UI demo, gọi API FastAPI
data/
  raw/        # tài liệu nguồn (sample.txt có sẵn để test)
  chroma_db/  # dữ liệu Chroma persist trên đĩa
evaluation/
  metrics.py run_eval.py eval_dataset/qa_pairs.json   # Recall@k, MRR
scripts/
  ingest_documents.py build_index.py       # ingest thủ công / ingest cả thư mục data/raw
  smoke_test_gemini.py                     # test nhanh kết nối embedding + generate
  smoke_test_pipeline.py                   # chạy thử full pipeline (ingest + query) trên sample.txt
  list_gemini_models.py                    # liệt kê model mà API key hiện có quyền dùng
tests/       # unit test (chunking, RRF, reranker)
```

Mỗi package trong `app/` có `__init__.py` re-export API công khai, ví dụ:
```python
from app.retrieval import VectorStore, Reranker, HybridRetriever
from app.query_processing import rewrite_query, decompose_query, expand_query
from app.pipeline import RagPipeline
```

## Setup

Dự án dùng virtualenv riêng (`.venv/`), đã cài sẵn toàn bộ dependency trong `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env   # rồi điền GEMINI_API_KEY (lấy tại https://aistudio.google.com/apikey)
```

Mọi lệnh chạy Python bên dưới đều dùng `.\.venv\Scripts\python.exe` (Windows) thay vì `python` hệ thống.

**Chạy script trong package luôn dùng `-m` từ thư mục gốc** (`python -m scripts.ten_file`), không chạy trực tiếp `python scripts\file.py` — vì cách sau không thêm project root vào import path, sẽ báo `ModuleNotFoundError: No module named 'app'`.

## Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload      # API tại http://localhost:8000/docs
.\.venv\Scripts\python.exe -m streamlit run ui\streamlit_app.py  # UI tại http://localhost:8501
```

## Ingest tài liệu

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest_documents data/raw/doc1.txt data/raw/doc2.pdf
# hoặc ingest cả thư mục data/raw/
.\.venv\Scripts\python.exe -m scripts.build_index
```

Hỗ trợ `.txt` và `.pdf` (đọc qua `pypdf`).

Ingest chỉ cần chạy **một lần** (dữ liệu persist trong `data/chroma_db/`) — server API/Streamlit khởi động sau đó sẽ tự động dựng lại BM25 sparse retriever từ dữ liệu đã lưu, không cần ingest lại mỗi lần chạy process mới.

Tất cả file trong `data/raw/` dùng chung 1 Chroma collection (`documents`). Nếu chỉ muốn truy vấn trên corpus thật của bạn (không lẫn `sample.txt` demo), xoá thư mục `data/chroma_db/` rồi ingest lại riêng file của bạn, hoặc đổi `CHROMA_COLLECTION_NAME` trong `.env`.

## Test & Eval

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evaluation.run_eval    # Recall@5, MRR trên evaluation/eval_dataset/qa_pairs.json
```

`qa_pairs.json` hiện có 4 câu hỏi mẫu ứng với nội dung `data/raw/sample.txt` — thay bằng câu hỏi/`relevant_chunk_ids` từ corpus thật của bạn khi làm báo cáo.

## Cấu hình (`.env`)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `GEMINI_API_KEY` | *(bắt buộc)* | Lấy tại Google AI Studio |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | |
| `LLM_MODEL` | `gemini-flash-lite-latest` | Xem lưu ý quota bên dưới |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Ký tự mỗi chunk |
| `TOP_K_DENSE` / `TOP_K_RERANK` | `20` / `5` | Số ứng viên truy xuất / số chunk cuối đưa vào LLM |
| `ENABLE_QUERY_EXPANSION` | `False` | Bật để demo ablation study; tăng đáng kể số lượt gọi LLM |
| `QUERY_EXPANSION_VARIANTS` | `2` | Số biến thể câu hỏi mỗi lần expansion |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder **đa ngôn ngữ** (có Việt) — xem lưu ý bên dưới |

### Lưu ý về reranker khi corpus/câu hỏi khác ngôn ngữ

Bản đầu dùng `cross-encoder/ms-marco-MiniLM-L-6-v2` (chỉ huấn luyện tiếng Anh). Khi test với corpus tiếng Anh (`FSoft_HR.pdf`) và câu hỏi tiếng Việt, reranker này chấm điểm sai lệch — đẩy các chunk tiếng Việt không liên quan lên trên dù bước hybrid retrieval đã tìm đúng chunk tiếng Anh liên quan. Đổi sang `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (huấn luyện trên mMARCO, có tiếng Việt) khắc phục được vấn đề này. Nếu corpus của bạn thuần một ngôn ngữ khớp với câu hỏi, có thể đổi lại reranker tiếng Anh gốc để nhẹ hơn.

### Lưu ý quan trọng về quota Gemini free tier

Model mới nhất (`gemini-flash-latest` → `gemini-3.6-flash` tại thời điểm viết) chỉ có **20 request/ngày** ở free tier — dễ hết quota chỉ sau vài lần test vì mỗi câu hỏi tốn nhiều lượt gọi LLM (rewrite + multi-hop decompose + generate, cộng thêm expand nếu bật). Vì vậy dự án mặc định dùng `gemini-flash-lite-latest`, có quota rộng rãi hơn nhiều. Muốn xem model nào key của bạn dùng được:

```powershell
.\.venv\Scripts\python.exe -m scripts.list_gemini_models
```

## Kiểm thử nhanh khi phát triển

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test_gemini     # chỉ test kết nối embedding + generate
.\.venv\Scripts\python.exe -m scripts.smoke_test_pipeline   # ingest sample.txt + chạy 1 câu hỏi qua toàn bộ pipeline
```
