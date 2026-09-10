from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.models.document import DocumentStatus
from app.services.hybrid_search import RetrievalMode


class IndexingResult(BaseModel):
    """文档向量索引结果。"""

    document_id: UUID
    status: DocumentStatus
    vector_count: int
    model_name: str
    sparse_model_name: str
    dimension: int
    elapsed_ms: int


class SemanticSearchRequest(BaseModel):
    """知识库语义检索参数。"""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("检索问题不能为空")
        return normalized


class SemanticSearchItem(BaseModel):
    """一条包含原文和来源的检索结果。"""

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    score: float
    page_number: int | None
    section_title: str | None
    vector_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)


class SemanticSearchResult(BaseModel):
    """知识库语义检索响应数据。"""

    query: str
    items: list[SemanticSearchItem]
    total: int
    model_name: str
    retrieval_mode: RetrievalMode
    reranker_applied: bool
    reranker_model: str | None


class RagAskRequest(BaseModel):
    """基于知识库生成诊断答案的参数。"""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    score_threshold: float | None = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("诊断问题不能为空")
        return normalized


class RagSource(BaseModel):
    """诊断答案引用的一条知识库原文。"""

    citation: str
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    score: float
    page_number: int | None
    section_title: str | None
    vector_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)


class LLMUsageRead(BaseModel):
    """本次生成的Token用量。"""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class RagAnswerResult(BaseModel):
    """带可追溯资料来源的RAG诊断结果。"""

    question: str
    answer: str
    sources: list[RagSource]
    retrieved_count: int
    embedding_model: str
    llm_model: str
    llm_called: bool
    usage: LLMUsageRead
    retrieval_mode: RetrievalMode
    reranker_applied: bool
    reranker_model: str | None
