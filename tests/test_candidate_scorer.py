from __future__ import annotations

from engine.candidate_gen import CandidateGenerator
from engine.scorer import RecommendationScorer


def test_candidate_generator_hybrid_filters_seen_and_duplicates() -> None:
    user_items = {1: [1, 2, 3], 2: [2, 4, 5], 3: [1, 6]}
    item_similarities = {
        1: [7, 8, 9],
        2: [8, 10],
        3: [11],
        4: [10],
        5: [12],
        6: [7],
    }
    item_popularity = {7: 20, 8: 18, 9: 16, 10: 30, 11: 25, 12: 12}
    item_categories = {
        1: "AI",
        2: "AI",
        3: "AI",
        4: "AI",
        5: "Web Dev",
        6: "AI",
        7: "AI",
        8: "AI",
        9: "Web Dev",
        10: "AI",
        11: "AI",
        12: "Data Science",
    }

    generator = CandidateGenerator(
        user_items=user_items,
        item_similarities=item_similarities,
        item_popularity=item_popularity,
        user_recent_items={1: [3, 2, 1]},
        item_categories=item_categories,
        user_preferred_category={1: "AI"},
        category_popularity={"AI": {10: 40, 7: 35, 8: 30}},
    )

    candidates = generator.hybrid_candidates(user_id=1, top_n=10)

    assert len(candidates) == len(set(candidates))
    assert set(candidates).isdisjoint({1, 2, 3})
    assert 10 in candidates


def test_candidate_generator_cold_start_uses_category_popularity() -> None:
    generator = CandidateGenerator(
        user_items={99: []},
        item_similarities={},
        item_popularity={1: 5, 2: 4, 3: 3},
        user_preferred_category={99: "Web Dev"},
        category_popularity={"Web Dev": {8: 25, 9: 20, 10: 15}},
    )

    popular = generator.popularity_candidates(top_n=3, user_id=99)

    assert popular == [8, 9, 10]
    assert generator.collaborative_candidates(99) == []
    assert generator.content_based_candidates(99) == []


def test_recommendation_scorer_dynamic_weights_and_scaling() -> None:
    scorer = RecommendationScorer()

    scorer.add_scorer("relevance", lambda _u, _i, _c: 1.2, weight=2.0)
    scorer.add_scorer("popularity", lambda _u, _i, _c: 0.5, weight=1.0)
    scorer.add_scorer("recency", lambda _u, _i, _c: -0.5, weight=0.5)

    score, explanation = scorer.calculate_score(
        user_id=1,
        item_id=42,
        context={"user_history_length": 5},
    )

    assert 0.0 <= score <= 1.0
    assert explanation["scores"]["relevance"] == 1.0
    assert explanation["scores"]["recency"] == 0.0
    assert explanation["weights"]["relevance"] > explanation["weights"]["popularity"]
    assert explanation["weights"]["popularity"] > explanation["weights"]["recency"]


def test_recommendation_scorer_rank_candidates_sorted() -> None:
    scorer = RecommendationScorer()
    scorer.add_scorer("relevance", lambda _u, item_id, _c: item_id / 10, weight=2.0)
    scorer.add_scorer("popularity", lambda _u, item_id, _c: (10 - item_id) / 10, weight=1.0)

    ranked = scorer.rank_candidates(user_id=1, candidates=[1, 5, 9], limit=2)

    assert len(ranked) == 2
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_recommendation_scorer_handles_no_scorers() -> None:
    scorer = RecommendationScorer()

    score, explanation = scorer.calculate_score(user_id=1, item_id=1)

    assert score == 0.0
    assert explanation["message"] == "No scorers registered"


def test_recommendation_scorer_remove_scorer() -> None:
    scorer = RecommendationScorer()
    scorer.add_scorer("temp", lambda _u, _i, _c: 0.5, weight=1.0)

    removed = scorer.remove_scorer("temp")
    not_removed = scorer.remove_scorer("temp")

    assert removed is True
    assert not_removed is False
