"""
Recommendation Evaluator Module

This module provides a RecommendationEvaluator class for measuring
the quality of recommendation systems using standard metrics.
"""

import math
from typing import Dict, List, Set


class RecommendationEvaluator:
    """
    A class to evaluate recommendation system performance.
    
    Provides standard evaluation metrics including Precision@K, Recall@K,
    and NDCG@K for measuring recommendation quality.
    """
    
    def precision_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Measures the proportion of recommended items in top-k that are relevant.
        
        Formula: (# of relevant items in top-k) / k
        
        Args:
            recommendations: Ordered list of recommended item IDs.
            relevant_items: Set of actually relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Float between 0 and 1 representing precision.
            
        Example:
            >>> evaluator = RecommendationEvaluator()
            >>> recs = [1, 2, 3, 4, 5]
            >>> relevant = {1, 3, 5, 7}
            >>> evaluator.precision_at_k(recs, relevant, k=3)
            0.6667
        """
        # Handle edge cases
        if k <= 0:
            return 0.0
        
        if not recommendations or not relevant_items:
            return 0.0
        
        # Get top-k recommendations
        top_k = recommendations[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
        
        # Calculate precision
        return relevant_in_top_k / k
    
    def recall_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Measures the proportion of relevant items that appear in top-k recommendations.
        
        Formula: (# of relevant items in top-k) / (total # of relevant items)
        
        Args:
            recommendations: Ordered list of recommended item IDs.
            relevant_items: Set of actually relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Float between 0 and 1 representing recall.
            
        Example:
            >>> evaluator = RecommendationEvaluator()
            >>> recs = [1, 2, 3, 4, 5]
            >>> relevant = {1, 3, 5, 7}
            >>> evaluator.recall_at_k(recs, relevant, k=5)
            0.75
        """
        # Handle edge cases
        if k <= 0:
            return 0.0
        
        if not recommendations:
            return 0.0
        
        if not relevant_items:
            return 0.0  # No relevant items to recall
        
        # Get top-k recommendations
        top_k = recommendations[:k]
        
        # Count relevant items in top-k
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
        
        # Calculate recall
        return relevant_in_top_k / len(relevant_items)
    
    def _dcg_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain at K.
        
        DCG accounts for the position of relevant items - items ranked higher
        contribute more to the score.
        
        Formula: sum(rel_i / log2(i + 1)) for i in 1 to k
        
        Args:
            recommendations: Ordered list of recommended item IDs.
            relevant_items: Set of actually relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            DCG score (unnormalized).
        """
        if k <= 0 or not recommendations:
            return 0.0
        
        top_k = recommendations[:k]
        dcg = 0.0
        
        for i, item in enumerate(top_k):
            # Relevance is binary: 1 if relevant, 0 otherwise
            rel = 1.0 if item in relevant_items else 0.0
            
            # Position is 1-indexed, so i+1
            # Discount factor: log2(position + 1)
            discount = math.log2(i + 2)  # i+2 because i is 0-indexed
            
            dcg += rel / discount
        
        return dcg
    
    def _idcg_at_k(
        self,
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Calculate Ideal Discounted Cumulative Gain at K.
        
        IDCG is the maximum possible DCG, achieved when all top-k items
        are relevant, ordered by relevance.
        
        Args:
            relevant_items: Set of actually relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            IDCG score (maximum possible DCG).
        """
        if k <= 0:
            return 0.0
        
        # Ideal case: all positions up to min(k, num_relevant) have relevant items
        num_relevant = len(relevant_items)
        ideal_k = min(k, num_relevant)
        
        idcg = 0.0
        for i in range(ideal_k):
            # All items are relevant in ideal ranking
            discount = math.log2(i + 2)  # i+2 because i is 0-indexed
            idcg += 1.0 / discount
        
        return idcg
    
    def ndcg_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG normalizes DCG by the ideal DCG, giving a score between 0 and 1.
        Higher positions contribute more to the score, rewarding good ranking.
        
        Formula: DCG@K / IDCG@K
        
        Args:
            recommendations: Ordered list of recommended item IDs.
            relevant_items: Set of actually relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Float between 0 and 1 representing normalized DCG.
            
        Example:
            >>> evaluator = RecommendationEvaluator()
            >>> recs = [1, 2, 3, 4, 5]  # 1 and 3 are relevant
            >>> relevant = {1, 3}
            >>> evaluator.ndcg_at_k(recs, relevant, k=5)
            0.8154
        """
        # Handle edge cases
        if k <= 0:
            return 0.0
        
        if not recommendations or not relevant_items:
            return 0.0
        
        # Calculate DCG and IDCG
        dcg = self._dcg_at_k(recommendations, relevant_items, k)
        idcg = self._idcg_at_k(relevant_items, k)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_all(
        self,
        recommendations_dict: Dict[int, List[int]],
        ground_truth_dict: Dict[int, List[int]],
        k: int
    ) -> Dict[str, float]:
        """
        Evaluate recommendations across all users.
        
        Computes average Precision@K, Recall@K, and NDCG@K across all users.
        
        Args:
            recommendations_dict: Dict mapping user_id to list of recommended item IDs.
            ground_truth_dict: Dict mapping user_id to list of actual relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Dictionary with average metrics:
            - 'precision@k': Average precision across users.
            - 'recall@k': Average recall across users.
            - 'ndcg@k': Average NDCG across users.
            - 'num_users': Number of users evaluated.
            
        Example:
            >>> metrics = evaluator.evaluate_all(recs_dict, truth_dict, k=10)
            >>> print(f"Precision@10: {metrics['precision@k']:.4f}")
        """
        # Handle edge cases
        if not recommendations_dict or not ground_truth_dict:
            return {
                f"precision@{k}": 0.0,
                f"recall@{k}": 0.0,
                f"ndcg@{k}": 0.0,
                "num_users": 0
            }
        
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        
        # Find common users between recommendations and ground truth
        common_users = set(recommendations_dict.keys()) & set(ground_truth_dict.keys())
        
        for user_id in common_users:
            recs = recommendations_dict.get(user_id, [])
            relevant = set(ground_truth_dict.get(user_id, []))
            
            # Skip users with no ground truth
            if not relevant:
                continue
            
            # Calculate metrics for this user
            precision = self.precision_at_k(recs, relevant, k)
            recall = self.recall_at_k(recs, relevant, k)
            ndcg = self.ndcg_at_k(recs, relevant, k)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
        
        # Calculate averages
        num_users = len(precision_scores)
        
        if num_users == 0:
            return {
                f"precision@{k}": 0.0,
                f"recall@{k}": 0.0,
                f"ndcg@{k}": 0.0,
                "num_users": 0
            }
        
        return {
            f"precision@{k}": sum(precision_scores) / num_users,
            f"recall@{k}": sum(recall_scores) / num_users,
            f"ndcg@{k}": sum(ndcg_scores) / num_users,
            "num_users": num_users
        }


if __name__ == "__main__":
    print("=" * 50)
    print("RecommendationEvaluator - Example Test Cases")
    print("=" * 50)
    
    evaluator = RecommendationEvaluator()
    
    # Test Precision@K
    print("\n1. Precision@K Tests:")
    print("-" * 30)
    
    # Recommendations: [1, 2, 3, 4, 5], Relevant: {1, 3, 5, 7}
    recs = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5, 7}
    
    p_at_3 = evaluator.precision_at_k(recs, relevant, k=3)
    print(f"   Recommendations: {recs}")
    print(f"   Relevant items: {relevant}")
    print(f"   Precision@3: {p_at_3:.4f}")
    print(f"   (2 relevant in top 3: items 1, 3)")
    
    p_at_5 = evaluator.precision_at_k(recs, relevant, k=5)
    print(f"\n   Precision@5: {p_at_5:.4f}")
    print(f"   (3 relevant in top 5: items 1, 3, 5)")
    
    # Test Recall@K
    print("\n2. Recall@K Tests:")
    print("-" * 30)
    
    r_at_3 = evaluator.recall_at_k(recs, relevant, k=3)
    print(f"   Recommendations: {recs}")
    print(f"   Relevant items: {relevant} (total: 4)")
    print(f"   Recall@3: {r_at_3:.4f}")
    print(f"   (2 out of 4 relevant items found)")
    
    r_at_5 = evaluator.recall_at_k(recs, relevant, k=5)
    print(f"\n   Recall@5: {r_at_5:.4f}")
    print(f"   (3 out of 4 relevant items found)")
    
    # Test NDCG@K
    print("\n3. NDCG@K Tests:")
    print("-" * 30)
    
    # Perfect ranking: relevant items at top
    perfect_recs = [1, 3, 5, 7, 2]  # All relevant items first
    relevant_2 = {1, 3, 5, 7}
    
    ndcg_perfect = evaluator.ndcg_at_k(perfect_recs, relevant_2, k=5)
    print(f"   Perfect ranking: {perfect_recs}")
    print(f"   Relevant items: {relevant_2}")
    print(f"   NDCG@5 (perfect): {ndcg_perfect:.4f}")
    
    # Suboptimal ranking
    suboptimal_recs = [2, 1, 4, 3, 5]  # Relevant items mixed in
    ndcg_suboptimal = evaluator.ndcg_at_k(suboptimal_recs, relevant_2, k=5)
    print(f"\n   Suboptimal ranking: {suboptimal_recs}")
    print(f"   NDCG@5 (suboptimal): {ndcg_suboptimal:.4f}")
    
    # Worst case: relevant items at bottom
    worst_recs = [10, 11, 12, 1, 3]  # Relevant items at end
    ndcg_worst = evaluator.ndcg_at_k(worst_recs, relevant_2, k=5)
    print(f"\n   Worst ranking: {worst_recs}")
    print(f"   NDCG@5 (worst): {ndcg_worst:.4f}")
    
    # Test evaluate_all
    print("\n4. Evaluate All Users Test:")
    print("-" * 30)
    
    # Sample recommendations for multiple users
    recommendations_dict = {
        1: [101, 102, 103, 104, 105],
        2: [201, 202, 203, 204, 205],
        3: [301, 302, 303, 304, 305],
        4: [401, 402, 403, 404, 405]
    }
    
    # Ground truth: items users actually liked
    ground_truth_dict = {
        1: [101, 103, 107],       # 2/3 relevant in top 5
        2: [202, 204, 206, 208],  # 2/4 relevant in top 5
        3: [301, 302, 303],       # 3/3 relevant in top 5 (perfect)
        4: [410, 411, 412]        # 0/3 relevant in top 5 (poor)
    }
    
    print("   Recommendations per user:")
    for user_id, recs in recommendations_dict.items():
        truth = ground_truth_dict.get(user_id, [])
        print(f"     User {user_id}: {recs}")
        print(f"             Ground truth: {truth}")
    
    metrics = evaluator.evaluate_all(recommendations_dict, ground_truth_dict, k=5)
    
    print(f"\n   Average Metrics (K=5):")
    print(f"     Precision@5: {metrics['precision@5']:.4f}")
    print(f"     Recall@5: {metrics['recall@5']:.4f}")
    print(f"     NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"     Users evaluated: {metrics['num_users']}")
    
    # Test edge cases
    print("\n5. Edge Case Tests:")
    print("-" * 30)
    
    # Empty recommendations
    empty_p = evaluator.precision_at_k([], {1, 2, 3}, k=5)
    print(f"   Empty recommendations: Precision = {empty_p:.4f}")
    
    # No relevant items
    no_rel_p = evaluator.precision_at_k([1, 2, 3], set(), k=3)
    print(f"   No relevant items: Precision = {no_rel_p:.4f}")
    
    # K = 0
    k_zero = evaluator.precision_at_k([1, 2, 3], {1, 2}, k=0)
    print(f"   K = 0: Precision = {k_zero:.4f}")
    
    # Missing user in evaluate_all
    partial_recs = {1: [1, 2, 3]}
    partial_truth = {1: [1], 2: [4, 5]}  # User 2 has no recommendations
    partial_metrics = evaluator.evaluate_all(partial_recs, partial_truth, k=3)
    print(f"   Missing user: Users evaluated = {partial_metrics['num_users']}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
