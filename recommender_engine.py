"""
Recommender Engine Module

This module integrates all recommendation components into a unified engine:
- SimilarityCalculator: For measuring user/item similarities
- CandidateGenerator: For generating recommendation candidates
- RecommendationScorer: For scoring and ranking candidates

Usage:
    engine = RecommenderEngine(user_items, item_similarities, item_popularity)
    recommendations = engine.recommend(user_id=1, limit=5)
"""

from typing import Any, Dict, List

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer


class RecommenderEngine:
    """
    A unified recommendation engine that combines candidate generation,
    scoring, and ranking into a single easy-to-use interface.
    
    This class integrates:
    - CandidateGenerator: Finds potential items to recommend
    - RecommendationScorer: Scores and ranks candidates
    - SimilarityCalculator: Available for similarity computations
    
    Attributes:
        user_items: Dict mapping user_id to list of item_ids they've interacted with.
        item_similarities: Dict mapping item_id to list of similar item_ids.
        item_popularity: Dict mapping item_id to popularity score.
        candidate_generator: Instance of CandidateGenerator.
        scorer: Instance of RecommendationScorer.
        similarity_calculator: Instance of SimilarityCalculator.
    """
    
    def __init__(
        self,
        user_items: Dict[int, List[int]],
        item_similarities: Dict[int, List[int]],
        item_popularity: Dict[int, float]
    ):
        """
        Initialize the RecommenderEngine with data dictionaries.
        
        Args:
            user_items: Dict mapping user_id to list of item_ids.
            item_similarities: Dict mapping item_id to list of similar item_ids.
            item_popularity: Dict mapping item_id to popularity score.
        """
        # Store the data
        self.user_items = user_items
        self.item_similarities = item_similarities
        self.item_popularity = item_popularity
        
        # Calculate max popularity for normalization
        self.max_popularity = max(item_popularity.values()) if item_popularity else 1.0
        
        # Initialize the candidate generator
        self.candidate_generator = CandidateGenerator(
            user_items=user_items,
            item_similarities=item_similarities,
            item_popularity=item_popularity
        )
        
        # Initialize the scorer
        self.scorer = RecommendationScorer()
        
        # Initialize the similarity calculator (available for external use)
        self.similarity_calculator = SimilarityCalculator()
        
        # Register default scoring functions with weights
        self._register_default_scorers()
    
    def _register_default_scorers(self) -> None:
        """
        Register the default scoring functions with the scorer.
        
        Default scorers:
        - relevance_score (weight: 2.0) - How relevant is the item to the user
        - popularity_score (weight: 1.0) - How popular is the item
        - recency_score (weight: 0.5) - How new/recent is the item
        """
        # Relevance has highest weight - most important factor
        self.scorer.add_scorer(
            name="relevance",
            function=self._relevance_score,
            weight=2.0
        )
        
        # Popularity is secondary
        self.scorer.add_scorer(
            name="popularity",
            function=self._popularity_score,
            weight=1.0
        )
        
        # Recency has lowest weight
        self.scorer.add_scorer(
            name="recency",
            function=self._recency_score,
            weight=0.5
        )
    
    def _relevance_score(
        self,
        user_id: int,
        item_id: int,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance score based on item similarity to user's history.
        
        The score is higher if the item is similar to items the user has
        previously interacted with.
        
        Args:
            user_id: The user's ID.
            item_id: The item's ID.
            context: Additional context (unused here).
            
        Returns:
            Float between 0 and 1 representing relevance.
        """
        # Get user's item history
        user_history = set(self.user_items.get(user_id, []))
        
        # If user has no history, return neutral score
        if not user_history:
            return 0.5
        
        # Check how many of the user's items are similar to this item
        similar_items = set(self.item_similarities.get(item_id, []))
        
        # If the item has no similarity data, check reverse
        if not similar_items:
            # Check if any user item lists this item as similar
            for user_item in user_history:
                if item_id in self.item_similarities.get(user_item, []):
                    return 0.8  # High relevance if directly similar
            return 0.5  # Neutral if no similarity info
        
        # Calculate overlap between user history and items similar to this one
        overlap = len(user_history.intersection(similar_items))
        
        if overlap > 0:
            # Normalize by user history size (more overlap = higher score)
            return min(1.0, 0.5 + (overlap / len(user_history)) * 0.5)
        
        return 0.3  # Low relevance if no overlap
    
    def _popularity_score(
        self,
        user_id: int,
        item_id: int,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate popularity score based on item's overall popularity.
        
        More popular items get higher scores.
        
        Args:
            user_id: The user's ID (unused here).
            item_id: The item's ID.
            context: Additional context (unused here).
            
        Returns:
            Float between 0 and 1 representing popularity.
        """
        # Get item's popularity score
        popularity = self.item_popularity.get(item_id, 0)
        
        # Normalize to [0, 1] range
        if self.max_popularity > 0:
            return popularity / self.max_popularity
        
        return 0.0
    
    def _recency_score(
        self,
        user_id: int,
        item_id: int,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate recency score based on item freshness.
        
        This is a simplified implementation. In a real system, you would
        use actual timestamps. Here we use item_id as a proxy (higher = newer).
        
        Args:
            user_id: The user's ID (unused here).
            item_id: The item's ID.
            context: Additional context - can contain 'item_ages' dict.
            
        Returns:
            Float between 0 and 1 representing recency.
        """
        # Check if item ages are provided in context
        item_ages = context.get("item_ages", {})
        
        if item_ages:
            # Use provided age data (lower age = newer = higher score)
            max_age = context.get("max_age", 365)
            age = item_ages.get(item_id, max_age)
            return max(0.0, 1.0 - (age / max_age))
        
        # Fallback: use item_id as proxy for recency
        # Higher item IDs are assumed to be newer
        max_item_id = max(self.item_popularity.keys()) if self.item_popularity else 10
        return min(1.0, item_id / max_item_id)
    
    def recommend(
        self,
        user_id: int,
        limit: int = 5,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user using the full pipeline.
        
        Pipeline Steps:
        1. Generate candidate items using hybrid_candidates
        2. Score each candidate using registered scoring functions
        3. Rank candidates by their final score
        4. Return the top N recommendations
        
        Args:
            user_id: The user to generate recommendations for.
            limit: Maximum number of recommendations to return (default: 5).
            context: Optional additional context for scoring functions.
            
        Returns:
            List of recommendation dicts with 'item_id', 'score', and 'explanation'.
            
        Example:
            >>> engine = RecommenderEngine(user_items, item_sim, item_pop)
            >>> recs = engine.recommend(user_id=1, limit=5)
            >>> for rec in recs:
            ...     print(f"Item {rec['item_id']}: {rec['score']:.2f}")
        """
        if context is None:
            context = {}
        
        # Step 1: Generate candidate items
        # Use hybrid approach to get diverse candidates
        candidates = self.candidate_generator.hybrid_candidates(
            user_id=user_id,
            top_n=limit * 3  # Get more candidates than needed for better ranking
        )
        
        # Handle case with no candidates
        if not candidates:
            return []
        
        # Step 2 & 3: Score and rank candidates
        # The scorer handles both scoring and ranking
        ranked_items = self.scorer.rank_candidates(
            user_id=user_id,
            candidates=candidates,
            context=context,
            limit=limit
        )
        
        # Step 4: Return top N recommendations
        return ranked_items
    
    def add_custom_scorer(
        self,
        name: str,
        function: callable,
        weight: float = 1.0
    ) -> None:
        """
        Add a custom scoring function to the engine.
        
        This allows extending the scoring with domain-specific logic.
        
        Args:
            name: Unique name for the scorer.
            function: Scoring function (user_id, item_id, context) -> float.
            weight: Weight for this scorer (default: 1.0).
            
        Example:
            >>> def seasonal_boost(user_id, item_id, context):
            ...     seasonal_items = context.get('seasonal', set())
            ...     return 1.0 if item_id in seasonal_items else 0.0
            >>> engine.add_custom_scorer("seasonal", seasonal_boost, weight=0.5)
        """
        self.scorer.add_scorer(name, function, weight)
    
    def get_user_history(self, user_id: int) -> List[int]:
        """
        Get a user's interaction history.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            List of item_ids the user has interacted with.
        """
        return self.user_items.get(user_id, [])
    
    def get_similar_items(self, item_id: int) -> List[int]:
        """
        Get items similar to a given item.
        
        Args:
            item_id: The item's ID.
            
        Returns:
            List of similar item_ids.
        """
        return self.item_similarities.get(item_id, [])


if __name__ == "__main__":
    # ==========================================================================
    # EXAMPLE: Using the RecommenderEngine
    # ==========================================================================
    
    print("=" * 60)
    print(" RecommenderEngine - Demo")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Create sample data
    # -------------------------------------------------------------------------
    print("\n1. Setting up sample data...")
    print("-" * 40)
    
    # User interaction history
    # - User 1 likes action/sci-fi movies (items 1, 2, 3)
    # - User 2 likes comedies (items 2, 3, 4, 5)
    # - User 3 likes dramas (items 1, 5, 6)
    # - User 4 likes thrillers (items 7, 8, 9)
    # - User 5 is a new user (no history)
    user_items = {
        1: [1, 2, 3],
        2: [2, 3, 4, 5],
        3: [1, 5, 6],
        4: [7, 8, 9],
        5: []  # New user - cold start
    }
    
    # Item similarities (content-based)
    item_similarities = {
        1: [2, 7],      # Item 1 is similar to items 2, 7
        2: [1, 3],      # Item 2 is similar to items 1, 3
        3: [2, 4],      # Item 3 is similar to items 2, 4
        4: [3, 5],      # Item 4 is similar to items 3, 5
        5: [4, 6],      # Item 5 is similar to items 4, 6
        6: [5, 10],     # Item 6 is similar to items 5, 10
        7: [1, 8],      # Item 7 is similar to items 1, 8
        8: [7, 9],      # Item 8 is similar to items 7, 9
        9: [8, 10],     # Item 9 is similar to items 8, 10
        10: [6, 9]      # Item 10 is similar to items 6, 9
    }
    
    # Item popularity scores (0-100)
    item_popularity = {
        1: 95,   # Very popular
        2: 85,
        3: 70,
        4: 90,   # Very popular
        5: 60,
        6: 50,
        7: 40,
        8: 88,   # Popular
        9: 30,
        10: 75
    }
    
    print("   Users: 5 (including 1 cold start user)")
    print("   Items: 10")
    print("   User-Item Interactions: ", sum(len(v) for v in user_items.values()))
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize the engine
    # -------------------------------------------------------------------------
    print("\n2. Initializing RecommenderEngine...")
    print("-" * 40)
    
    engine = RecommenderEngine(
        user_items=user_items,
        item_similarities=item_similarities,
        item_popularity=item_popularity
    )
    
    print("   ✓ CandidateGenerator initialized")
    print("   ✓ RecommendationScorer initialized")
    print("   ✓ Default scorers registered:")
    print("     - relevance (weight: 2.0)")
    print("     - popularity (weight: 1.0)")
    print("     - recency (weight: 0.5)")
    
    # -------------------------------------------------------------------------
    # Step 3: Generate recommendations for different users
    # -------------------------------------------------------------------------
    print("\n3. Generating Recommendations...")
    print("-" * 40)
    
    # Recommend for User 1 (has history)
    print("\n   User 1 (history: [1, 2, 3]):")
    recommendations = engine.recommend(user_id=1, limit=5)
    
    for i, rec in enumerate(recommendations, 1):
        item_id = rec["item_id"]
        score = rec["score"]
        print(f"     #{i}: Item {item_id:2d} | Score: {score:.4f}")
    
    # Recommend for User 4 (different preferences)
    print("\n   User 4 (history: [7, 8, 9]):")
    recommendations = engine.recommend(user_id=4, limit=5)
    
    for i, rec in enumerate(recommendations, 1):
        item_id = rec["item_id"]
        score = rec["score"]
        print(f"     #{i}: Item {item_id:2d} | Score: {score:.4f}")
    
    # Recommend for User 5 (cold start - no history)
    print("\n   User 5 (cold start - no history):")
    recommendations = engine.recommend(user_id=5, limit=5)
    
    for i, rec in enumerate(recommendations, 1):
        item_id = rec["item_id"]
        score = rec["score"]
        print(f"     #{i}: Item {item_id:2d} | Score: {score:.4f}")
    print("     (Falls back to popularity-based recommendations)")
    
    # -------------------------------------------------------------------------
    # Step 4: Show detailed explanation for one recommendation
    # -------------------------------------------------------------------------
    print("\n4. Detailed Score Breakdown...")
    print("-" * 40)
    
    print("\n   User 1, Top Recommendation:")
    if recommendations := engine.recommend(user_id=1, limit=1):
        rec = recommendations[0]
        explanation = rec["explanation"]
        
        print(f"     Item: {rec['item_id']}")
        print(f"     Final Score: {rec['score']:.4f}")
        print("     Score Components:")
        
        for scorer_name in explanation["scores"]:
            raw = explanation["scores"][scorer_name]
            weight = explanation["weights"][scorer_name]
            weighted = explanation["weighted_scores"][scorer_name]
            print(f"       - {scorer_name}: {raw:.2f} × {weight:.1f} = {weighted:.2f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Demonstrate adding a custom scorer
    # -------------------------------------------------------------------------
    print("\n5. Adding Custom Scorer...")
    print("-" * 40)
    
    # Define a custom "promotion" scorer that boosts certain items
    def promotion_score(user_id, item_id, context):
        """Boost items that are on promotion."""
        promoted_items = {4, 8, 10}  # Items on sale
        return 1.0 if item_id in promoted_items else 0.0
    
    engine.add_custom_scorer("promotion", promotion_score, weight=1.5)
    print("   ✓ Added 'promotion' scorer (weight: 1.5)")
    print("   Promoted items: {4, 8, 10}")
    
    print("\n   User 1 recommendations (with promotion boost):")
    recommendations = engine.recommend(user_id=1, limit=5)
    
    for i, rec in enumerate(recommendations, 1):
        item_id = rec["item_id"]
        score = rec["score"]
        promoted = "⭐" if item_id in {4, 8, 10} else "  "
        print(f"     #{i}: Item {item_id:2d} | Score: {score:.4f} {promoted}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60)
    print("""
 The RecommenderEngine successfully:
 ✓ Generated candidates using hybrid approach
 ✓ Scored candidates with multiple weighted factors
 ✓ Ranked and returned top recommendations
 ✓ Handled cold-start users with popularity fallback
 ✓ Supported custom scoring functions
""")
