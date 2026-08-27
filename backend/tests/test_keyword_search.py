from uuid import uuid4

from app.services.hybrid_search import reciprocal_rank_fusion
from app.services.keyword_search import KeywordSearchHit, bm25_scores, tokenize_for_bm25
from app.services.semantic_search import SemanticSearchHit


def test_chinese_bm25_tokenizer_preserves_fault_codes_and_bigrams() -> None:
    tokens = tokenize_for_bm25("GA-55空压机出现E101冷却器高温")

    assert "ga-55" in tokens
    assert "e101" in tokens
    assert "冷却" in tokens
    assert "却器" in tokens
    assert "高温" in tokens


def test_bm25_ranks_exact_fault_code_first() -> None:
    corpus = [
        tokenize_for_bm25("E101表示空压机排气温度过高"),
        tokenize_for_bm25("E201表示电机过载"),
        tokenize_for_bm25("日常检查空气过滤器"),
    ]

    scores = bm25_scores(tokenize_for_bm25("E101故障"), corpus)

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_rrf_merges_vector_and_keyword_rankings() -> None:
    shared_chunk = uuid4()
    vector_only_chunk = uuid4()
    document_id = uuid4()
    vector_hits = [
        SemanticSearchHit(
            chunk_id=vector_only_chunk,
            document_id=document_id,
            filename="manual.md",
            content="语义相似内容",
            score=0.9,
            page_number=1,
            section_title="语义章节",
            vector_score=0.9,
            retrieval_sources=["vector"],
        ),
        SemanticSearchHit(
            chunk_id=shared_chunk,
            document_id=document_id,
            filename="manual.md",
            content="E101高温停机",
            score=0.8,
            page_number=2,
            section_title="E101",
            vector_score=0.8,
            retrieval_sources=["vector"],
        ),
    ]
    keyword_hits = [
        KeywordSearchHit(
            chunk_id=shared_chunk,
            document_id=document_id,
            filename="manual.md",
            content="E101高温停机",
            score=3.2,
            page_number=2,
            section_title="E101",
        )
    ]

    results = reciprocal_rank_fusion(vector_hits, keyword_hits)

    assert results[0].chunk_id == shared_chunk
    assert results[0].retrieval_sources == ["vector", "keyword"]
    assert results[0].vector_score == 0.8
    assert results[0].keyword_score == 3.2
    assert results[0].fusion_score == 1.0
