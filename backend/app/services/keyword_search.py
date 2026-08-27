import math
import re
from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import document as repository

LATIN_OR_NUMBER_PATTERN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.IGNORECASE)
CHINESE_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True, slots=True)
class KeywordSearchHit:
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    score: float
    page_number: int | None
    section_title: str | None


def tokenize_for_bm25(text: str) -> list[str]:
    """保留故障代码等英文数字串，并为连续中文生成单字和二元词。"""
    normalized = text.lower()
    tokens = LATIN_OR_NUMBER_PATTERN.findall(normalized)
    for segment in CHINESE_PATTERN.findall(normalized):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def bm25_scores(
    query_tokens: list[str],
    corpus_tokens: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """计算Okapi BM25分数，空查询或空语料直接返回零分。"""
    if not corpus_tokens:
        return []
    if not query_tokens:
        return [0.0] * len(corpus_tokens)

    document_count = len(corpus_tokens)
    average_length = sum(map(len, corpus_tokens)) / document_count or 1.0
    document_frequencies: Counter[str] = Counter()
    for tokens in corpus_tokens:
        document_frequencies.update(set(tokens))

    scores: list[float] = []
    for tokens in corpus_tokens:
        frequencies = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term in set(query_tokens):
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * document_length / average_length)
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


async def search_keywords(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    limit: int,
) -> list[KeywordSearchHit]:
    """在指定知识库的SQL切片上执行BM25关键词检索。"""
    records = await repository.get_knowledge_base_chunks_with_documents(
        session,
        knowledge_base_id,
    )
    if not records:
        return []

    query_tokens = tokenize_for_bm25(query)
    corpus_tokens = [tokenize_for_bm25(chunk.content) for chunk, _ in records]
    scores = bm25_scores(query_tokens, corpus_tokens)
    ranked_indexes = sorted(
        range(len(records)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results: list[KeywordSearchHit] = []
    for index in ranked_indexes:
        score = scores[index]
        if score <= 0:
            continue
        chunk, document = records[index]
        results.append(
            KeywordSearchHit(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                content=chunk.content,
                score=round(score, 6),
                page_number=chunk.page_number,
                section_title=chunk.extra_metadata.get("section_title"),
            )
        )
        if len(results) == limit:
            break
    return results
