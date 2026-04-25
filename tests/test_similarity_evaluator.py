import pytest

from engine.evaluator import RecommendationEvaluator
from engine.similarity import SimilarityCalculator


def test_similarity_calculator_metrics() -> None:
    calc = SimilarityCalculator()

    assert calc.cosine_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    assert calc.jaccard_similarity({1, 2, 3}, {2, 3, 4}) == 0.5
    assert calc.pearson_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_similarity_calculator_edge_cases() -> None:
    calc = SimilarityCalculator()

    assert calc.cosine_similarity([1], [1, 2]) == 0.0
    assert calc.jaccard_similarity(set(), set()) == 0.0
    assert calc.pearson_correlation([3, 3, 3], [1, 2, 3]) == 0.0


def test_recommendation_evaluator_metrics() -> None:
    evaluator = RecommendationEvaluator()
    recs = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6}

    assert evaluator.precision_at_k(recs, relevant, k=5) == 0.4
    assert evaluator.recall_at_k(recs, relevant, k=5) == (2 / 3)
    ndcg = evaluator.ndcg_at_k(recs, relevant, k=5)
    assert 0.0 <= ndcg <= 1.0


def test_recommendation_evaluator_evaluate_all() -> None:
    evaluator = RecommendationEvaluator()
    recommendations = {1: [1, 2, 3], 2: [4, 5, 6]}
    ground_truth = {1: [1, 4], 2: [5]}

    metrics = evaluator.evaluate_all(recommendations, ground_truth, k=3)

    assert "precision@3" in metrics
    assert "recall@3" in metrics
    assert "ndcg@3" in metrics
    assert metrics["num_users"] == 2
