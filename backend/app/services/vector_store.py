from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.services.sparse_embedding import SparseEmbedding

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class VectorStoreError(RuntimeError):
    """Qdrant集合配置或读写操作失败。"""


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """准备写入Qdrant的向量点。"""

    point_id: str
    dense_vector: list[float]
    sparse_vector: SparseEmbedding
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    """Qdrant返回的向量命中。"""

    point_id: str
    score: float
    payload: dict[str, Any]
    vector_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    retrieval_sources: tuple[str, ...] = ()


class QdrantVectorStore:
    """封装本地模式和远程Qdrant共同使用的操作。"""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name

    def ensure_collection(self, dimension: int) -> None:
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=dimension,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    },
                )
                return

            collection = self.client.get_collection(self.collection_name)
            vector_config = collection.config.params.vectors
            sparse_config = collection.config.params.sparse_vectors
            if not isinstance(vector_config, dict) or DENSE_VECTOR_NAME not in vector_config:
                raise VectorStoreError(
                    "Qdrant集合仍是旧版单向量结构，请更换QDRANT_COLLECTION名称并重新索引文档"
                )
            if not isinstance(sparse_config, dict) or SPARSE_VECTOR_NAME not in sparse_config:
                raise VectorStoreError("Qdrant集合缺少sparse命名向量，请重新建立索引")
            if vector_config[DENSE_VECTOR_NAME].size != dimension:
                raise VectorStoreError(
                    f"Qdrant集合维度为{vector_config[DENSE_VECTOR_NAME].size}，"
                    f"当前模型维度为{dimension}"
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("无法创建或检查Qdrant集合") from exc

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point.point_id,
                        vector={
                            DENSE_VECTOR_NAME: point.dense_vector,
                            SPARSE_VECTOR_NAME: models.SparseVector(
                                indices=point.sparse_vector.indices,
                                values=point.sparse_vector.values,
                            ),
                        },
                        payload=point.payload,
                    )
                    for point in points
                ],
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("向Qdrant写入向量失败") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            if not self.client.collection_exists(self.collection_name):
                return
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("删除文档向量失败") from exc

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        """删除一个知识库下的全部向量，防止SQL级联删除后留下孤立数据。"""
        try:
            if not self.client.collection_exists(self.collection_name):
                return
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="knowledge_base_id",
                                match=models.MatchValue(value=knowledge_base_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("删除知识库向量失败") from exc

    def search(
        self,
        query_vector: list[float],
        knowledge_base_id: str,
        limit: int,
        score_threshold: float | None,
    ) -> list[VectorSearchHit]:
        try:
            if not self.client.collection_exists(self.collection_name):
                return []
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="knowledge_base_id",
                            match=models.MatchValue(value=knowledge_base_id),
                        )
                    ]
                ),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            return [
                VectorSearchHit(
                    point_id=str(point.id),
                    score=float(point.score),
                    payload=point.payload or {},
                    vector_score=float(point.score),
                    retrieval_sources=("vector",),
                )
                for point in response.points
            ]
        except Exception as exc:
            raise VectorStoreError("Qdrant语义检索失败") from exc

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseEmbedding,
        knowledge_base_id: str,
        limit: int,
        score_threshold: float | None,
    ) -> list[VectorSearchHit]:
        """由Qdrant原生执行Dense/Sparse召回和RRF融合。"""
        try:
            if not self.client.collection_exists(self.collection_name):
                return []
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="knowledge_base_id",
                        match=models.MatchValue(value=knowledge_base_id),
                    )
                ]
            )
            sparse_query = models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            )
            dense_prefetch = models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
            sparse_prefetch = models.Prefetch(
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=limit,
            )
            dense_request = models.QueryRequest(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
            sparse_request = models.QueryRequest(
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=limit,
            )
            fused_request = models.QueryRequest(
                prefetch=[dense_prefetch, sparse_prefetch],
                query=models.RrfQuery(rrf=models.Rrf(k=max(settings.rrf_k, 1))),
                limit=limit,
                with_payload=True,
            )
            dense_response, sparse_response, fused_response = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=[dense_request, sparse_request, fused_request],
            )
            dense_scores = {str(point.id): float(point.score) for point in dense_response.points}
            sparse_scores = {str(point.id): float(point.score) for point in sparse_response.points}
            return [
                VectorSearchHit(
                    point_id=str(point.id),
                    score=float(point.score),
                    payload=point.payload or {},
                    vector_score=dense_scores.get(str(point.id)),
                    sparse_score=sparse_scores.get(str(point.id)),
                    fusion_score=float(point.score),
                    retrieval_sources=tuple(
                        source
                        for source, scores in (
                            ("vector", dense_scores),
                            ("keyword", sparse_scores),
                        )
                        if str(point.id) in scores
                    ),
                )
                for point in fused_response.points
            ]
        except Exception as exc:
            raise VectorStoreError("Qdrant混合检索失败") from exc


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    """根据配置连接远程Qdrant或打开本地持久化模式。"""
    if settings.qdrant_url:
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )
        client = QdrantClient(url=settings.qdrant_url, api_key=api_key)
    else:
        client = QdrantClient(path=str(settings.qdrant_path))
    return QdrantVectorStore(client, settings.qdrant_collection)
