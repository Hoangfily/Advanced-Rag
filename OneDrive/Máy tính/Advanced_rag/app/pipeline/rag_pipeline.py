from app.core.config import get_settings
from app.generation.llm_client import LLMClient
from app.generation.prompt_templates import RAG_ANSWER_TEMPLATE
from app.ingestion.chunking import chunk_text
from app.ingestion.embedder import Embedder
from app.ingestion.loaders import load_documents
from app.query_processing.expansion import expand_query
from app.query_processing.multi_hop import decompose_query
from app.query_processing.rewriter import rewrite_query
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.models import QueryResponse, RetrievedChunk


class RagPipeline:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model, settings.gemini_api_key)
        self.llm_client = LLMClient(settings.llm_model, settings.gemini_api_key)
        self.vector_store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection_name, self.embedder)
        self.reranker = Reranker(settings.reranker_model)
        self._sparse_retriever: SparseRetriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._rebuild_retrievers_from_store()

    def _rebuild_retrievers_from_store(self) -> None:
        existing_chunks = self.vector_store.get_all_chunks()
        if not existing_chunks:
            return
        self._sparse_retriever = SparseRetriever(existing_chunks, top_k=self.settings.top_k_sparse)
        self._hybrid_retriever = HybridRetriever(
            self.vector_store, self._sparse_retriever, top_k_dense=self.settings.top_k_dense
        )

    def ingest(self, paths: list[str]) -> int:
        documents = load_documents(paths)
        all_chunks = []
        for source, text in documents.items():
            all_chunks.extend(chunk_text(text, source, self.settings.chunk_size, self.settings.chunk_overlap))

        self.vector_store.add_chunks(all_chunks)

        self._rebuild_retrievers_from_store()
        return len(all_chunks)

    def run(self, question: str, top_k: int | None = None) -> QueryResponse:
        top_k = top_k or self.settings.top_k_rerank

        rewritten = rewrite_query(question, self.llm_client)
        sub_queries = decompose_query(rewritten, self.llm_client)

        candidates: list[RetrievedChunk] = []
        for sub_query in sub_queries:
            query_variants = [sub_query]
            if self.settings.enable_query_expansion:
                query_variants += expand_query(
                    sub_query, self.llm_client, self.settings.query_expansion_variants
                )

            for variant in query_variants:
                candidates.extend(self._hybrid_retriever.retrieve(variant, self.settings.top_k_dense))

        reranked = self.reranker.rerank(rewritten, candidates, top_k)
        context = "\n\n".join(c.text for c in reranked)
        prompt = RAG_ANSWER_TEMPLATE.format(context=context, question=question)
        answer = self.llm_client.complete(prompt)

        return QueryResponse(
            answer=answer,
            sub_queries=sub_queries,
            contexts=reranked,
        )
