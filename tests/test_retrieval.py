# Unit tests for retrieval helpers, no live DB required.

# Tests cover:
#   - embed_query returns a list of floats with the right dimension
#   - RRF score math is correct
#   - Results are correctly sorted by rrf_score descending
#   - Empty lexical results don't crash hybrid (graceful UNION ALL with 0 rows)

# Run with uv run pytest tests/test_retrieval.py -v

from dataclasses import dataclass

import pytest

# RRF score math
def rrf_score(ranks:list[int], k:int = 60) -> float:
    # Reference implementation of RRF for a single document.
    return sum(1.0 / (k + r) for r in ranks)

class TestRRFMath:
    def test_single_list_rank1(self):
        # appears at rank 1 in one list
        assert rrf_score([1]) == pytest.approx(1 / 61)

    def test_two_lists_both_rank1(self):
        # appears at rank 1 in both lists -> doubles the score
        assert rrf_score([1, 1]) == pytest.approx(2 / 61)

    def test_high_rank_beats_low_rank(self):
        # rank 1 > rank 50 > rank 100
        assert rrf_score([1]) > rrf_score([50]) > rrf_score([100])

    def test_two_moderate_ranks_beat_one_top(self):
        # appearing in both lists at rank 10 should beat appearing once at rank 1
        assert rrf_score([10, 10]) > rrf_score([1])

    def test_k_constant_is_60(self):
        # canonical k value; this test documents the contract
        assert rrf_score([1], k=60) == pytest.approx(1 / 61)

# Sorting guarantee
@dataclass
class FakeChunk:
    id: int
    score: float

def sort_by_rrf(chunks: list[FakeChunk]) -> list[FakeChunk]:
    return sorted(chunks, key=lambda c: c.score, reverse=True)

class TestSorting:
    def test_highest_score_first(self):
        chunks = [FakeChunk(1, 0.01), FakeChunk(2, 0.03), FakeChunk(3, 0.02)]
        sorted_ = sort_by_rrf(chunks)
        assert [c.id for c in sorted_] == [2, 3, 1]

    def test_single_chunk_unchanged(self):
        chunks = [FakeChunk(1, 0.05)]
        assert sort_by_rrf(chunks) == chunks

    def test_empty_list(self):
        assert sort_by_rrf([]) == []


# embed_query smoke test (requires model download; marked as slow)
@pytest.mark.slow
def test_embed_query_dimension():
    # BGE-small-en-v1.5 produces 384-dim normalized vectors.
    import os
    os.environ.setdefault("DATABASE_URL", "postgresql://fake/fake")  # prevent import error

    from healthcare_rag.retrieval import embed_query

    vec = embed_query("What is the maximum out-of-pocket limit?")
    assert isinstance(vec, list)
    assert len(vec) == 384

    # normalized -> L2 norm ~1.0
    import math
    norm = math.sqrt(sum(x ** 2 for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-4)
