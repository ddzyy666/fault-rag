from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.hybrid_search import RetrievalMode


class RetrievalEvaluationCase(BaseModel):
    """一条带有人工相关性标注的检索评估问题。"""

    case_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    relevant_chunk_ids: list[UUID] = Field(min_length=1, max_length=20)

    @field_validator("case_id", "query")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("评估用例编号和问题不能为空")
        return normalized

    @field_validator("relevant_chunk_ids")
    @classmethod
    def unique_relevant_chunks(cls, value: list[UUID]) -> list[UUID]:
        if len(set(value)) != len(value):
            raise ValueError("同一用例不能重复标注相关切片")
        return value


class RetrievalEvaluationVariant(BaseModel):
    """一组需要参与对比的检索配置。"""

    name: str = Field(min_length=1, max_length=50)
    retrieval_mode: RetrievalMode
    rerank: bool

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("评估方案名称不能为空")
        return normalized


def default_retrieval_variants() -> list[RetrievalEvaluationVariant]:
    return [
        RetrievalEvaluationVariant(
            name="vector",
            retrieval_mode=RetrievalMode.VECTOR,
            rerank=False,
        ),
        RetrievalEvaluationVariant(
            name="hybrid_rrf",
            retrieval_mode=RetrievalMode.HYBRID,
            rerank=False,
        ),
        RetrievalEvaluationVariant(
            name="hybrid_rerank",
            retrieval_mode=RetrievalMode.HYBRID,
            rerank=True,
        ),
    ]


class RetrievalEvaluationRequest(BaseModel):
    """批量检索效果评估请求。"""

    cases: list[RetrievalEvaluationCase] = Field(min_length=1, max_length=50)
    variants: list[RetrievalEvaluationVariant] = Field(
        default_factory=default_retrieval_variants,
        min_length=1,
        max_length=5,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_unique_names_and_workload(self) -> "RetrievalEvaluationRequest":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("评估用例case_id不能重复")
        variant_names = [variant.name for variant in self.variants]
        if len(set(variant_names)) != len(variant_names):
            raise ValueError("评估方案name不能重复")
        if len(self.cases) * len(self.variants) > 100:
            raise ValueError("单次评估最多执行100次检索")
        return self


class RetrievalEvaluationCaseResult(BaseModel):
    """单条问题在某一检索方案下的评估明细。"""

    case_id: str
    query: str
    relevant_chunk_ids: list[UUID]
    retrieved_chunk_ids: list[UUID]
    relevant_ranks: list[int]
    hit: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: int
    reranker_applied: bool


class RetrievalEvaluationMetrics(BaseModel):
    """一组检索配置的聚合指标。"""

    case_count: int
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    average_latency_ms: float
    p95_latency_ms: int
    reranker_applied_count: int


class RetrievalEvaluationVariantResult(BaseModel):
    """一组检索配置的完整评估结果。"""

    name: str
    retrieval_mode: RetrievalMode
    rerank: bool
    metrics: RetrievalEvaluationMetrics
    cases: list[RetrievalEvaluationCaseResult]


class RetrievalEvaluationResult(BaseModel):
    """多组检索方案对比结果。"""

    knowledge_base_id: UUID
    top_k: int
    score_threshold: float | None
    best_variant: str
    variants: list[RetrievalEvaluationVariantResult]
