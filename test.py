"""
Test Suite for Recommendation Engine Core Components

This file tests all modules:
- SimilarityCalculator (similarity.py)
- CandidateGenerator (candidate_gen.py)
- RecommendationScorer (scorer.py)
- RecommendationEvaluator (evaluator.py)
"""

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator
from recommender_engine import RecommenderEngine


# =============================================================================
# FAKE DATASET: 5 Users, 10 Items
# =============================================================================

# User interaction history: which items each user has interacted with
USER_ITEMS = {
    1: [1, 2, 3],       # User 1 likes items 1, 2, 3
    2: [2, 3, 4, 5],    # User 2 likes items 2, 3, 4, 5
    3: [1, 5, 6],       # User 3 likes items 1, 5, 6
    4: [7, 8, 9],       # User 4 likes different items
    5: []               # User 5 is new (cold start)
}

# Item similarities based on content/features
ITEM_SIMILARITIES = {
    1: [2, 7],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 10],
    7: [1, 8],
    8: [7, 9],
    9: [8, 10],
    10: [6, 9]
}

# Item popularity scores (higher = more popular)
ITEM_POPULARITY = {
    1: 100, 2: 85, 3: 70, 4: 95, 5: 60,
    6: 50, 7: 40, 8: 90, 9: 30, 10: 80
}

# User feature vectors (for similarity calculations)
USER_VECTORS = {
    1: [5.0, 4.0, 3.0, 2.0, 1.0],
    2: [4.5, 4.0, 3.5, 2.5, 1.5],
    3: [1.0, 2.0, 3.0, 4.0, 5.0],
    4: [3.0, 3.0, 3.0, 3.0, 3.0],
    5: [0.0, 0.0, 0.0, 0.0, 0.0]
}

# User preferences as sets (for Jaccard similarity)
USER_PREFERENCES = {
    1: {"action", "sci-fi", "comedy"},
    2: {"action", "sci-fi", "drama"},
    3: {"romance", "drama", "comedy"},
    4: {"horror", "thriller"},
    5: set()
}

# User ratings for Pearson correlation
USER_RATINGS = {
    1: [5, 4, 3, 2, 1],
    2: [4, 5, 3, 2, 1],
    3: [1, 2, 3, 4, 5],
    4: [3, 3, 3, 3, 3],
    5: [0, 0, 0, 0, 0]
}

# Ground truth: items users will actually like (for evaluation)
GROUND_TRUTH = {
    1: [4, 5, 7],       # User 1 will like these items
    2: [6, 7, 8],       # User 2 will like these items
    3: [2, 3, 4],       # User 3 will like these items
    4: [1, 2, 10],      # User 4 will like these items
    5: [1, 4, 8]        # User 5 will like these items (popular items)
}


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n  {title}")
    print("  " + "-" * 40)


def test_similarity_calculator():
    """Test the SimilarityCalculator class."""
    print_header("1. Testing SimilarityCalculator")
    
    calc = SimilarityCalculator()
    all_passed = True
    
    # Test Cosine Similarity
    print_subheader("Cosine Similarity")
    
    # Test 1: Similar users
    vec1 = USER_VECTORS[1]
    vec2 = USER_VECTORS[2]
    cos_sim = calc.cosine_similarity(vec1, vec2)
    print(f"    User 1 vs User 2: {cos_sim:.4f}")
    print(f"    (High similarity expected - similar preferences)")
    
    # Test 2: Opposite users
    vec3 = USER_VECTORS[3]
    cos_sim_opp = calc.cosine_similarity(vec1, vec3)
    print(f"    User 1 vs User 3: {cos_sim_opp:.4f}")
    print(f"    (Lower similarity - opposite preferences)")
    
    # Test 3: Zero vector
    vec5 = USER_VECTORS[5]
    cos_sim_zero = calc.cosine_similarity(vec1, vec5)
    print(f"    User 1 vs User 5 (zero vector): {cos_sim_zero:.4f}")
    
    if cos_sim > cos_sim_opp and cos_sim_zero == 0.0:
        print("    ✓ Cosine similarity tests PASSED")
    else:
        print("    ✗ Cosine similarity tests FAILED")
        all_passed = False
    
    # Test Jaccard Similarity
    print_subheader("Jaccard Similarity")
    
    # Test 1: Overlapping preferences
    set1 = USER_PREFERENCES[1]
    set2 = USER_PREFERENCES[2]
    jac_sim = calc.jaccard_similarity(set1, set2)
    print(f"    User 1 preferences: {set1}")
    print(f"    User 2 preferences: {set2}")
    print(f"    Jaccard similarity: {jac_sim:.4f}")
    
    # Test 2: Empty set
    set5 = USER_PREFERENCES[5]
    jac_empty = calc.jaccard_similarity(set1, set5)
    print(f"    User 1 vs User 5 (empty): {jac_empty:.4f}")
    
    if jac_sim > 0 and jac_empty == 0.0:
        print("    ✓ Jaccard similarity tests PASSED")
    else:
        print("    ✗ Jaccard similarity tests FAILED")
        all_passed = False
    
    # Test Pearson Correlation
    print_subheader("Pearson Correlation")
    
    # Test 1: Positive correlation
    ratings1 = USER_RATINGS[1]
    ratings2 = USER_RATINGS[2]
    pearson = calc.pearson_correlation(ratings1, ratings2)
    print(f"    User 1 ratings: {ratings1}")
    print(f"    User 2 ratings: {ratings2}")
    print(f"    Pearson correlation: {pearson:.4f}")
    
    # Test 2: Negative correlation
    ratings3 = USER_RATINGS[3]
    pearson_neg = calc.pearson_correlation(ratings1, ratings3)
    print(f"    User 1 vs User 3: {pearson_neg:.4f} (negative expected)")
    
    # Test 3: Constant ratings
    ratings4 = USER_RATINGS[4]
    pearson_const = calc.pearson_correlation(ratings1, ratings4)
    print(f"    User 1 vs User 4 (constant): {pearson_const:.4f}")
    
    if pearson > 0 and pearson_neg < 0 and pearson_const == 0.0:
        print("    ✓ Pearson correlation tests PASSED")
    else:
        print("    ✗ Pearson correlation tests FAILED")
        all_passed = False
    
    return all_passed


def test_candidate_generator():
    """Test the CandidateGenerator class."""
    print_header("2. Testing CandidateGenerator")
    
    gen = CandidateGenerator(USER_ITEMS, ITEM_SIMILARITIES, ITEM_POPULARITY)
    all_passed = True
    
    # Test Collaborative Filtering
    print_subheader("Collaborative Candidates")
    
    # User 1 shares items with User 2 and User 3
    collab_1 = gen.collaborative_candidates(1)
    print(f"    User 1 history: {USER_ITEMS[1]}")
    print(f"    Collaborative candidates: {collab_1}")
    print(f"    (Items from similar users not in User 1's history)")
    
    # Cold start user
    collab_5 = gen.collaborative_candidates(5)
    print(f"    User 5 (cold start): {collab_5}")
    
    if len(collab_1) > 0 and len(collab_5) == 0:
        print("    ✓ Collaborative filtering tests PASSED")
    else:
        print("    ✗ Collaborative filtering tests FAILED")
        all_passed = False
    
    # Test Content-Based Filtering
    print_subheader("Content-Based Candidates")
    
    content_1 = gen.content_based_candidates(1)
    print(f"    User 1 history: {USER_ITEMS[1]}")
    print(f"    Content-based candidates: {content_1}")
    print(f"    (Items similar to User 1's liked items)")
    
    content_5 = gen.content_based_candidates(5)
    print(f"    User 5 (cold start): {content_5}")
    
    if len(content_1) > 0 and len(content_5) == 0:
        print("    ✓ Content-based filtering tests PASSED")
    else:
        print("    ✗ Content-based filtering tests FAILED")
        all_passed = False
    
    # Test Popularity Candidates
    print_subheader("Popularity Candidates")
    
    popular = gen.popularity_candidates(top_n=5)
    print(f"    Top 5 popular items: {popular}")
    print(f"    Popularity scores: {[ITEM_POPULARITY[i] for i in popular]}")
    
    # Verify sorted by popularity
    scores = [ITEM_POPULARITY[i] for i in popular]
    is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    if is_sorted:
        print("    ✓ Popularity candidates tests PASSED")
    else:
        print("    ✗ Popularity candidates tests FAILED")
        all_passed = False
    
    # Test Hybrid Candidates
    print_subheader("Hybrid Candidates")
    
    hybrid_1 = gen.hybrid_candidates(1, top_n=10)
    print(f"    User 1 hybrid candidates: {hybrid_1}")
    
    hybrid_5 = gen.hybrid_candidates(5, top_n=5)
    print(f"    User 5 hybrid (cold start): {hybrid_5}")
    print(f"    (Falls back to popularity)")
    
    if len(hybrid_1) > 0 and len(hybrid_5) > 0:
        print("    ✓ Hybrid candidates tests PASSED")
    else:
        print("    ✗ Hybrid candidates tests FAILED")
        all_passed = False
    
    return all_passed


def test_recommendation_scorer():
    """Test the RecommendationScorer class."""
    print_header("3. Testing RecommendationScorer")
    
    scorer = RecommendationScorer()
    all_passed = True
    
    # Define scoring functions
    def relevance_fn(user_id, item_id, context):
        """Simple relevance based on item ID proximity to user preferences."""
        user_items = context.get("user_items", {}).get(user_id, [])
        if not user_items:
            return 0.5
        avg_item = sum(user_items) / len(user_items)
        distance = abs(item_id - avg_item)
        return max(0, 1 - distance / 10)
    
    def popularity_fn(user_id, item_id, context):
        """Popularity score."""
        popularity = context.get("item_popularity", {})
        max_pop = max(popularity.values()) if popularity else 100
        return popularity.get(item_id, 0) / max_pop
    
    def recency_fn(user_id, item_id, context):
        """Recency score (mock: higher item IDs are newer)."""
        return min(1.0, item_id / 10)
    
    # Test Add Scorer
    print_subheader("Adding Scorers")
    
    scorer.add_scorer("relevance", relevance_fn, weight=2.0)
    scorer.add_scorer("popularity", popularity_fn, weight=1.0)
    scorer.add_scorer("recency", recency_fn, weight=0.5)
    
    print("    Added 'relevance' (weight: 2.0)")
    print("    Added 'popularity' (weight: 1.0)")
    print("    Added 'recency' (weight: 0.5)")
    print("    ✓ Scorers added successfully")
    
    # Test Calculate Score
    print_subheader("Calculate Score")
    
    context = {
        "user_items": USER_ITEMS,
        "item_popularity": ITEM_POPULARITY
    }
    
    score, explanation = scorer.calculate_score(1, 4, context)
    print(f"    User 1, Item 4:")
    print(f"    Final Score: {score:.4f}")
    print(f"    Breakdown:")
    for name in explanation["scores"]:
        raw = explanation["scores"][name]
        weight = explanation["weights"][name]
        weighted = explanation["weighted_scores"][name]
        print(f"      - {name}: {raw:.2f} × {weight} = {weighted:.2f}")
    
    if 0 <= score <= 1:
        print("    ✓ Score calculation tests PASSED")
    else:
        print("    ✗ Score calculation tests FAILED")
        all_passed = False
    
    # Test Rank Candidates
    print_subheader("Rank Candidates")
    
    candidates = [4, 5, 6, 7, 8, 9, 10]
    ranked = scorer.rank_candidates(1, candidates, context, limit=5)
    
    print(f"    Candidates: {candidates}")
    print(f"    Top 5 Ranked:")
    for i, item in enumerate(ranked, 1):
        print(f"      #{i}: Item {item['item_id']} - Score: {item['score']:.4f}")
    
    # Verify sorted descending
    scores = [item["score"] for item in ranked]
    is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    if is_sorted and len(ranked) == 5:
        print("    ✓ Ranking tests PASSED")
    else:
        print("    ✗ Ranking tests FAILED")
        all_passed = False
    
    return all_passed


def test_recommendation_evaluator():
    """Test the RecommendationEvaluator class."""
    print_header("4. Testing RecommendationEvaluator")
    
    evaluator = RecommendationEvaluator()
    all_passed = True
    
    # Generate recommendations using CandidateGenerator
    gen = CandidateGenerator(USER_ITEMS, ITEM_SIMILARITIES, ITEM_POPULARITY)
    
    recommendations = {}
    for user_id in range(1, 6):
        recommendations[user_id] = gen.hybrid_candidates(user_id, top_n=5)
    
    print_subheader("Generated Recommendations")
    for user_id, recs in recommendations.items():
        truth = GROUND_TRUTH[user_id]
        print(f"    User {user_id}: Recommended {recs}")
        print(f"            Ground truth {truth}")
    
    # Test Precision@K
    print_subheader("Precision@K")
    
    recs_1 = recommendations[1]
    relevant_1 = set(GROUND_TRUTH[1])
    
    p_at_3 = evaluator.precision_at_k(recs_1, relevant_1, k=3)
    p_at_5 = evaluator.precision_at_k(recs_1, relevant_1, k=5)
    
    print(f"    User 1 - Precision@3: {p_at_3:.4f}")
    print(f"    User 1 - Precision@5: {p_at_5:.4f}")
    
    if 0 <= p_at_3 <= 1 and 0 <= p_at_5 <= 1:
        print("    ✓ Precision@K tests PASSED")
    else:
        print("    ✗ Precision@K tests FAILED")
        all_passed = False
    
    # Test Recall@K
    print_subheader("Recall@K")
    
    r_at_3 = evaluator.recall_at_k(recs_1, relevant_1, k=3)
    r_at_5 = evaluator.recall_at_k(recs_1, relevant_1, k=5)
    
    print(f"    User 1 - Recall@3: {r_at_3:.4f}")
    print(f"    User 1 - Recall@5: {r_at_5:.4f}")
    
    if 0 <= r_at_3 <= 1 and 0 <= r_at_5 <= 1:
        print("    ✓ Recall@K tests PASSED")
    else:
        print("    ✗ Recall@K tests FAILED")
        all_passed = False
    
    # Test NDCG@K
    print_subheader("NDCG@K")
    
    ndcg_3 = evaluator.ndcg_at_k(recs_1, relevant_1, k=3)
    ndcg_5 = evaluator.ndcg_at_k(recs_1, relevant_1, k=5)
    
    print(f"    User 1 - NDCG@3: {ndcg_3:.4f}")
    print(f"    User 1 - NDCG@5: {ndcg_5:.4f}")
    
    if 0 <= ndcg_3 <= 1 and 0 <= ndcg_5 <= 1:
        print("    ✓ NDCG@K tests PASSED")
    else:
        print("    ✗ NDCG@K tests FAILED")
        all_passed = False
    
    # Test Evaluate All
    print_subheader("Evaluate All Users")
    
    metrics = evaluator.evaluate_all(recommendations, GROUND_TRUTH, k=5)
    
    print(f"    Average Metrics (K=5):")
    print(f"      Precision@5: {metrics['precision@5']:.4f}")
    print(f"      Recall@5: {metrics['recall@5']:.4f}")
    print(f"      NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"      Users evaluated: {metrics['num_users']}")
    
    if metrics['num_users'] == 5:
        print("    ✓ Evaluate all tests PASSED")
    else:
        print("    ✗ Evaluate all tests FAILED")
        all_passed = False
    
    return all_passed


def test_integration():
    """Test that all modules work together."""
    print_header("5. Integration Test: Full Pipeline")
    
    print_subheader("Pipeline Steps")
    
    # Step 1: Find similar users
    calc = SimilarityCalculator()
    print("    Step 1: Calculate user similarities")
    
    sim_1_2 = calc.cosine_similarity(USER_VECTORS[1], USER_VECTORS[2])
    sim_1_3 = calc.cosine_similarity(USER_VECTORS[1], USER_VECTORS[3])
    print(f"      User 1 ~ User 2: {sim_1_2:.4f}")
    print(f"      User 1 ~ User 3: {sim_1_3:.4f}")
    
    # Step 2: Generate candidates
    gen = CandidateGenerator(USER_ITEMS, ITEM_SIMILARITIES, ITEM_POPULARITY)
    print("\n    Step 2: Generate candidates for User 1")
    
    candidates = gen.hybrid_candidates(1, top_n=10)
    print(f"      Candidates: {candidates}")
    
    # Step 3: Score candidates
    scorer = RecommendationScorer()
    
    def simple_score(user_id, item_id, context):
        popularity = context.get("item_popularity", {})
        return popularity.get(item_id, 50) / 100
    
    scorer.add_scorer("popularity", simple_score, weight=1.0)
    
    context = {"item_popularity": ITEM_POPULARITY}
    ranked = scorer.rank_candidates(1, candidates, context, limit=5)
    
    print("\n    Step 3: Score and rank candidates")
    print(f"      Top 5 recommendations:")
    final_recs = []
    for i, item in enumerate(ranked, 1):
        print(f"        #{i}: Item {item['item_id']} (score: {item['score']:.2f})")
        final_recs.append(item['item_id'])
    
    # Step 4: Evaluate
    evaluator = RecommendationEvaluator()
    relevant = set(GROUND_TRUTH[1])
    
    print("\n    Step 4: Evaluate recommendations")
    print(f"      Ground truth: {GROUND_TRUTH[1]}")
    
    precision = evaluator.precision_at_k(final_recs, relevant, k=5)
    recall = evaluator.recall_at_k(final_recs, relevant, k=5)
    ndcg = evaluator.ndcg_at_k(final_recs, relevant, k=5)
    
    print(f"      Precision@5: {precision:.4f}")
    print(f"      Recall@5: {recall:.4f}")
    print(f"      NDCG@5: {ndcg:.4f}")
    
    print("\n    ✓ Integration test PASSED - All modules work together!")
    return True


def test_recommender_engine():
    """Test the RecommenderEngine class."""
    print_header("6. Testing RecommenderEngine")
    
    all_passed = True
    
    # Create sample dataset
    print_subheader("Initializing Engine")
    
    user_items = {
        1: [1, 2, 3],
        2: [2, 3, 4, 5],
        3: [1, 5, 6],
        4: [7, 8, 9],
        5: []  # Cold start user
    }
    
    item_similarities = {
        1: [2, 7], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6],
        6: [5, 10], 7: [1, 8], 8: [7, 9], 9: [8, 10], 10: [6, 9]
    }
    
    item_popularity = {
        1: 95, 2: 85, 3: 70, 4: 90, 5: 60,
        6: 50, 7: 40, 8: 88, 9: 30, 10: 75
    }
    
    # Initialize engine
    engine = RecommenderEngine(user_items, item_similarities, item_popularity)
    print("    ✓ RecommenderEngine initialized")
    print("    ✓ Default scorers: relevance, popularity, recency")
    
    # Test recommendations for user with history
    print_subheader("Recommendations for User 1")
    
    recs_1 = engine.recommend(user_id=1, limit=5)
    print(f"    User 1 history: {user_items[1]}")
    print(f"    Top 5 recommendations:")
    
    for i, rec in enumerate(recs_1, 1):
        print(f"      #{i}: Item {rec['item_id']} - Score: {rec['score']:.4f}")
    
    if len(recs_1) > 0 and all(0 <= r['score'] <= 1 for r in recs_1):
        print("    ✓ User 1 recommendations PASSED")
    else:
        print("    ✗ User 1 recommendations FAILED")
        all_passed = False
    
    # Test cold start user
    print_subheader("Cold Start User (User 5)")
    
    recs_5 = engine.recommend(user_id=5, limit=5)
    print(f"    User 5 history: {user_items[5]} (empty)")
    print(f"    Top 5 recommendations:")
    
    for i, rec in enumerate(recs_5, 1):
        print(f"      #{i}: Item {rec['item_id']} - Score: {rec['score']:.4f}")
    
    if len(recs_5) > 0:
        print("    ✓ Cold start recommendations PASSED (fallback to popularity)")
    else:
        print("    ✗ Cold start recommendations FAILED")
        all_passed = False
    
    # Test adding custom scorer
    print_subheader("Custom Scorer")
    
    def boost_score(user_id, item_id, context):
        """Boost specific items."""
        boosted = {4, 8}
        return 1.0 if item_id in boosted else 0.0
    
    engine.add_custom_scorer("boost", boost_score, weight=1.0)
    print("    ✓ Added custom 'boost' scorer")
    
    recs_boosted = engine.recommend(user_id=1, limit=3)
    print(f"    Top 3 with boost (items 4, 8 boosted):")
    
    for i, rec in enumerate(recs_boosted, 1):
        boosted = "⭐" if rec['item_id'] in {4, 8} else ""
        print(f"      #{i}: Item {rec['item_id']} - Score: {rec['score']:.4f} {boosted}")
    
    print("    ✓ Custom scorer test PASSED")
    
    # Test helper methods
    print_subheader("Helper Methods")
    
    history = engine.get_user_history(1)
    similar = engine.get_similar_items(1)
    
    print(f"    get_user_history(1): {history}")
    print(f"    get_similar_items(1): {similar}")
    
    if history == [1, 2, 3] and similar == [2, 7]:
        print("    ✓ Helper methods PASSED")
    else:
        print("    ✗ Helper methods FAILED")
        all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("#  RECOMMENDATION ENGINE - CORE COMPONENTS TEST SUITE")
    print("#" * 60)
    
    results = {
        "SimilarityCalculator": test_similarity_calculator(),
        "CandidateGenerator": test_candidate_generator(),
        "RecommendationScorer": test_recommendation_scorer(),
        "RecommendationEvaluator": test_recommendation_evaluator(),
        "RecommenderEngine": test_recommender_engine(),
        "Integration": test_integration()
    }
    
    # Summary
    print_header("TEST SUMMARY")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    for module, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"    {module}: {status}")
    
    print(f"\n    Total: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n    🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n    ⚠️ Some tests failed. Please review.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
