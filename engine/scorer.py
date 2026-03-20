"""
Recommendation Scorer Module

This module provides a RecommendationScorer class for scoring
recommendation candidates using weighted scoring functions.
"""

from typing import Any, Callable, Dict, List, Tuple


class RecommendationScorer:
    """
    A class to score recommendation candidates using configurable scoring functions.
    
    Allows registering multiple scoring functions with weights and combines
    them into a final score for ranking recommendations.
    
    Attributes:
        scorers: Dictionary storing registered scoring functions and their weights.
    """
    
    def __init__(self):
        """Initialize the RecommendationScorer with empty scorers dictionary."""
        self.scorers: Dict[str, Dict[str, Any]] = {}
    
    def add_scorer(
        self,
        name: str,
        function: Callable[[int, int, Dict], float],
        weight: float = 1.0
    ) -> None:
        """
        Register a scoring function with a name and weight.
        
        Args:
            name: Unique identifier for the scorer.
            function: Scoring function that takes (user_id, item_id, context) 
                      and returns a score between 0 and 1.
            weight: Weight multiplier for this scorer (default: 1.0).
            
        Example:
            >>> scorer = RecommendationScorer()
            >>> scorer.add_scorer("relevance", relevance_fn, weight=2.0)
        """
        self.scorers[name] = {
            "function": function,
            "weight": weight
        }
    
    def remove_scorer(self, name: str) -> bool:
        """
        Remove a registered scoring function.
        
        Args:
            name: Name of the scorer to remove.
            
        Returns:
            True if removed, False if not found.
        """
        if name in self.scorers:
            del self.scorers[name]
            return True
        return False
    
    def calculate_score(
        self,
        user_id: int,
        item_id: int,
        context: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the combined score for a user-item pair.
        
        Runs all registered scoring functions, multiplies by weights,
        and combines into a normalized final score.
        
        Args:
            user_id: The user's ID.
            item_id: The item's ID.
            context: Optional dictionary with additional context data.
            
        Returns:
            Tuple of (final_score, explanation_dict) where:
            - final_score: Float between 0 and 1.
            - explanation_dict: Breakdown of individual scores.
            
        Example:
            >>> score, explanation = scorer.calculate_score(1, 42, context)
            >>> print(f"Score: {score:.2f}")
            >>> print(f"Breakdown: {explanation}")
        """
        if context is None:
            context = {}
        
        # Handle case with no scorers
        if not self.scorers:
            return 0.0, {"message": "No scorers registered"}
        
        explanation: Dict[str, Any] = {
            "user_id": user_id,
            "item_id": item_id,
            "scores": {},
            "weights": {},
            "weighted_scores": {}
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Run each scoring function
        for name, scorer_info in self.scorers.items():
            func = scorer_info["function"]
            weight = scorer_info["weight"]
            
            try:
                # Call the scoring function
                raw_score = func(user_id, item_id, context)
                
                # Clamp score to [0, 1]
                raw_score = max(0.0, min(1.0, raw_score))
                
                # Calculate weighted score
                weighted_score = raw_score * weight
                
                # Store in explanation
                explanation["scores"][name] = raw_score
                explanation["weights"][name] = weight
                explanation["weighted_scores"][name] = weighted_score
                
                # Accumulate
                total_weighted_score += weighted_score
                total_weight += weight
                
            except Exception as e:
                # Handle scorer errors gracefully
                explanation["scores"][name] = 0.0
                explanation["weights"][name] = weight
                explanation["weighted_scores"][name] = 0.0
                explanation[f"{name}_error"] = str(e)
        
        # Calculate final normalized score
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.0
        
        # Clamp final score to [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        explanation["total_weight"] = total_weight
        explanation["final_score"] = final_score
        
        return final_score, explanation
    
    def rank_candidates(
        self,
        user_id: int,
        candidates: List[int],
        context: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Score and rank a list of candidate items.
        
        Scores each candidate, sorts by score descending, and returns
        the top N items with their scores and explanations.
        
        Args:
            user_id: The user's ID.
            candidates: List of item_ids to score.
            context: Optional dictionary with additional context data.
            limit: Maximum number of items to return (default: 10).
            
        Returns:
            List of dicts with 'item_id', 'score', and 'explanation'.
            
        Example:
            >>> ranked = scorer.rank_candidates(1, [10, 20, 30], limit=5)
            >>> for item in ranked:
            ...     print(f"Item {item['item_id']}: {item['score']:.2f}")
        """
        if context is None:
            context = {}
        
        # Score each candidate
        scored_items: List[Dict[str, Any]] = []
        
        for item_id in candidates:
            score, explanation = self.calculate_score(user_id, item_id, context)
            scored_items.append({
                "item_id": item_id,
                "score": score,
                "explanation": explanation
            })
        
        # Sort by score descending
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N items
        return scored_items[:limit]


# Example scoring functions
def relevance_score(user_id: int, item_id: int, context: Dict) -> float:
    """
    Example: Calculate relevance based on user-item match.
    
    In a real system, this would use ML models or similarity metrics.
    Here we use a simple mock based on context data.
    """
    user_preferences = context.get("user_preferences", {})
    item_features = context.get("item_features", {})
    
    user_prefs = user_preferences.get(user_id, set())
    item_feats = item_features.get(item_id, set())
    
    if not user_prefs or not item_feats:
        return 0.5  # Default neutral score
    
    # Calculate overlap ratio
    overlap = len(user_prefs.intersection(item_feats))
    max_possible = max(len(user_prefs), len(item_feats))
    
    if max_possible == 0:
        return 0.5
    
    return overlap / max_possible


def popularity_score(user_id: int, item_id: int, context: Dict) -> float:
    """
    Example: Score based on item popularity.
    
    Returns normalized popularity score from context data.
    """
    item_popularity = context.get("item_popularity", {})
    max_popularity = context.get("max_popularity", 100)
    
    popularity = item_popularity.get(item_id, 0)
    
    # Normalize to [0, 1]
    if max_popularity > 0:
        return min(1.0, popularity / max_popularity)
    return 0.0


def recency_score(user_id: int, item_id: int, context: Dict) -> float:
    """
    Example: Score based on item recency.
    
    Newer items get higher scores. Uses days since creation.
    """
    item_ages = context.get("item_ages", {})  # Days since creation
    max_age = context.get("max_age", 365)  # Consider items up to 1 year old
    
    age = item_ages.get(item_id, max_age)
    
    # Newer items (lower age) get higher scores
    if max_age > 0:
        recency = 1.0 - (min(age, max_age) / max_age)
        return max(0.0, recency)
    return 0.5


if __name__ == "__main__":
    print("=" * 50)
    print("RecommendationScorer - Example Test Cases")
    print("=" * 50)
    
    # Create scorer instance
    scorer = RecommendationScorer()
    
    # Register scoring functions with weights
    print("\n1. Registering Scorers:")
    print("-" * 30)
    scorer.add_scorer("relevance", relevance_score, weight=2.0)
    scorer.add_scorer("popularity", popularity_score, weight=1.0)
    scorer.add_scorer("recency", recency_score, weight=1.5)
    print("   Added 'relevance' scorer (weight: 2.0)")
    print("   Added 'popularity' scorer (weight: 1.0)")
    print("   Added 'recency' scorer (weight: 1.5)")
    
    # Create sample context data
    context = {
        # User preferences (what categories/features they like)
        "user_preferences": {
            1: {"action", "sci-fi", "comedy"},
            2: {"romance", "drama"},
            3: {"horror", "thriller"}
        },
        # Item features
        "item_features": {
            101: {"action", "sci-fi"},
            102: {"romance", "comedy"},
            103: {"action", "thriller"},
            104: {"drama", "romance"},
            105: {"comedy", "family"}
        },
        # Item popularity scores
        "item_popularity": {
            101: 85,
            102: 60,
            103: 95,
            104: 40,
            105: 70
        },
        "max_popularity": 100,
        # Item ages in days
        "item_ages": {
            101: 30,   # 1 month old
            102: 180,  # 6 months old
            103: 7,    # 1 week old
            104: 365,  # 1 year old
            105: 60    # 2 months old
        },
        "max_age": 365
    }
    
    # Test calculate_score
    print("\n2. Calculate Score Tests:")
    print("-" * 30)
    
    user_id = 1
    item_id = 101
    score, explanation = scorer.calculate_score(user_id, item_id, context)
    
    print(f"   User {user_id}, Item {item_id}")
    print(f"   Final Score: {score:.4f}")
    print(f"   Score Breakdown:")
    for name, raw_score in explanation["scores"].items():
        weight = explanation["weights"][name]
        weighted = explanation["weighted_scores"][name]
        print(f"     - {name}: {raw_score:.2f} × {weight} = {weighted:.2f}")
    
    # Test with different user-item pair
    print()
    user_id = 2
    item_id = 104
    score, explanation = scorer.calculate_score(user_id, item_id, context)
    
    print(f"   User {user_id}, Item {item_id}")
    print(f"   Final Score: {score:.4f}")
    print(f"   Score Breakdown:")
    for name, raw_score in explanation["scores"].items():
        weight = explanation["weights"][name]
        weighted = explanation["weighted_scores"][name]
        print(f"     - {name}: {raw_score:.2f} × {weight} = {weighted:.2f}")
    
    # Test rank_candidates
    print("\n3. Rank Candidates Test:")
    print("-" * 30)
    
    candidates = [101, 102, 103, 104, 105]
    user_id = 1
    
    print(f"   User {user_id}")
    print(f"   Candidates: {candidates}")
    print()
    
    ranked = scorer.rank_candidates(user_id, candidates, context, limit=5)
    
    print("   Ranking Results:")
    for rank, item in enumerate(ranked, 1):
        print(f"   #{rank}: Item {item['item_id']} - Score: {item['score']:.4f}")
    
    # Test with new user (limited context)
    print("\n4. Cold Start User Test:")
    print("-" * 30)
    
    user_id = 99  # New user not in preferences
    ranked = scorer.rank_candidates(user_id, candidates, context, limit=3)
    
    print(f"   User {user_id} (new user)")
    print("   Ranking Results (falls back to popularity + recency):")
    for rank, item in enumerate(ranked, 1):
        print(f"   #{rank}: Item {item['item_id']} - Score: {item['score']:.4f}")
    
    # Test adding custom scorer
    print("\n5. Custom Scorer Test:")
    print("-" * 30)
    
    def custom_boost(user_id: int, item_id: int, context: Dict) -> float:
        """Boost certain items."""
        boosted_items = context.get("boosted_items", set())
        return 1.0 if item_id in boosted_items else 0.0
    
    scorer.add_scorer("boost", custom_boost, weight=0.5)
    print("   Added custom 'boost' scorer (weight: 0.5)")
    
    context["boosted_items"] = {102, 105}
    print(f"   Boosted items: {context['boosted_items']}")
    
    ranked = scorer.rank_candidates(1, candidates, context, limit=3)
    print("\n   New Ranking with Boost:")
    for rank, item in enumerate(ranked, 1):
        print(f"   #{rank}: Item {item['item_id']} - Score: {item['score']:.4f}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
