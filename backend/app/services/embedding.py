from functools import lru_cache
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

from app.core.config import settings


class EmbeddingError(RuntimeError):
    """Embedding模型加载或推理失败。"""


class EmbeddingProvider(Protocol):
    """文档索引和查询向量化共同依赖的抽象接口。"""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def normalize_vector(vector: np.ndarray) -> list[float]:
    """执行L2归一化，使余弦相似度计算保持稳定。"""
    numeric = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(numeric))
    if norm == 0:
        raise EmbeddingError("Embedding模型返回了零向量")
    return (numeric / norm).tolist()


class FastEmbedProvider:
    """基于ONNX Runtime的轻量中文Embedding实现。"""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        batch_size: int,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._model: TextEmbedding | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            try:
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    lazy_load=True,
                )
            except Exception as exc:
                raise EmbeddingError(f"无法加载Embedding模型：{self.model_name}") from exc
        return self._model

    def _validate_vectors(self, vectors: list[list[float]]) -> list[list[float]]:
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError(f"Embedding维度与配置不一致，期望{self.dimension}维")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = self._get_model().passage_embed(
                texts,
                batch_size=self._batch_size,
            )
            vectors = [normalize_vector(vector) for vector in embeddings]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("文档向量生成失败") from exc
        return self._validate_vectors(vectors)

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = next(
                iter(
                    self._get_model().query_embed(
                        text,
                        batch_size=1,
                    )
                )
            )
        except Exception as exc:
            raise EmbeddingError("查询向量生成失败") from exc
        return self._validate_vectors([normalize_vector(vector)])[0]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """延迟创建并复用Embedding模型，避免每次请求重新加载。"""
    return FastEmbedProvider(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
