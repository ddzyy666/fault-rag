import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.services.keyword_search import tokenize_for_bm25


class SparseEmbeddingError(RuntimeError):
    """稀疏词法向量生成失败。"""


@dataclass(frozen=True, slots=True)
class SparseEmbedding:
    """Qdrant稀疏向量所需的有序索引和值。"""

    indices: list[int]
    values: list[float]


class SparseEmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]: ...

    def embed_query(self, text: str) -> SparseEmbedding: ...


class HashedLexicalEmbeddingProvider:
    """将中英文词元稳定映射为稀疏维度，由Qdrant在查询时计算IDF。"""

    model_name = "fault-lexical-hash-v1"

    @staticmethod
    def _token_index(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _embed(self, text: str) -> SparseEmbedding:
        tokens = tokenize_for_bm25(text)
        if not tokens:
            return SparseEmbedding(indices=[], values=[])

        frequencies: Counter[int] = Counter(self._token_index(token) for token in tokens)
        indices = sorted(frequencies)
        # 次线性TF避免重复词主导点积；集合级IDF由Qdrant的Modifier.IDF提供。
        values = [1.0 + math.log(frequencies[index]) for index in indices]
        return SparseEmbedding(indices=indices, values=values)

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> SparseEmbedding:
        return self._embed(text)


@lru_cache
def get_sparse_embedding_provider() -> SparseEmbeddingProvider:
    return HashedLexicalEmbeddingProvider()
