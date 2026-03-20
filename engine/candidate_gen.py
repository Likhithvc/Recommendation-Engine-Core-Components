"""
Candidate Generation Module

This module provides a CandidateGenerator class for generating
recommendation candidates using various strategies.
"""

from typing import Dict, List, Set


class CandidateGenerator:
    """
    A class to generate recommendation candidates using multiple strategies.
    
    Supports collaborative filtering, content-based filtering, popularity-based
    recommendations, and hybrid approaches.
    
    Attributes:
        user_items: Dictionary mapping user_id to list of item_ids they've interacted with.
        item_similarities: Dictionary mapping item_id to list of similar item_ids.
        item_popularity: Dictionary mapping item_id to popularity score.
    """
    
    def __init__(
        self,
        user_items: Dict[int, List[int]],
        item_similarities: Dict[int, List[int]],
        item_popularity: Dict[int, float]
    ):
        """
        Initialize the CandidateGenerator with data dictionaries.
        
        Args:
            user_items: Dict mapping user_id to list of item_ids.
            item_similarities: Dict mapping item_id to list of similar item_ids.
            item_popularity: Dict mapping item_id to popularity score.
        """
        self.user_items = user_items
        self.item_similarities = item_similarities
        self.item_popularity = item_popularity
    
    def _get_user_history(self, user_id: int) -> Set[int]:
        """
        Get the set of items a user has already interacted with.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            Set of item_ids the user has seen.
        """
        return set(self.user_items.get(user_id, []))
    
    def collaborative_candidates(self, user_id: int) -> List[int]:
        """
        Generate candidates using collaborative filtering.
        
        Finds users who share items with the given user, then recommends
        items those similar users liked but the current user hasn't seen.
        
        Args:
            user_id: The user to generate recommendations for.
            
        Returns:
            List of recommended item_ids.
            
        Example:
            >>> gen = CandidateGenerator(user_items, item_sim, item_pop)
            >>> gen.collaborative_candidates(1)
            [5, 6, 7]
        """
        # Get current user's item history
        user_history = self._get_user_history(user_id)
        
        # Handle cold start - new user with no history
        if not user_history:
            return []
        
        # Find similar users (those who share at least one item)
        similar_users: List[int] = []
        for other_user_id, other_items in self.user_items.items():
            if other_user_id == user_id:
                continue
            
            other_items_set = set(other_items)
            # Check if there's overlap between users
            if user_history.intersection(other_items_set):
                similar_users.append(other_user_id)
        
        # Collect candidate items from similar users
        candidates: Set[int] = set()
        for similar_user_id in similar_users:
            similar_user_items = set(self.user_items.get(similar_user_id, []))
            # Add items the similar user has but current user hasn't seen
            new_items = similar_user_items - user_history
            candidates.update(new_items)
        
        return list(candidates)
    
    def content_based_candidates(self, user_id: int) -> List[int]:
        """
        Generate candidates using content-based filtering.
        
        Recommends items similar to items in the user's history,
        using the item_similarities dictionary.
        
        Args:
            user_id: The user to generate recommendations for.
            
        Returns:
            List of recommended item_ids.
            
        Example:
            >>> gen = CandidateGenerator(user_items, item_sim, item_pop)
            >>> gen.content_based_candidates(1)
            [4, 5, 6]
        """
        # Get current user's item history
        user_history = self._get_user_history(user_id)
        
        # Handle cold start - new user with no history
        if not user_history:
            return []
        
        # Find similar items for each item in user's history
        candidates: Set[int] = set()
        for item_id in user_history:
            similar_items = self.item_similarities.get(item_id, [])
            candidates.update(similar_items)
        
        # Remove items the user has already seen
        candidates = candidates - user_history
        
        return list(candidates)
    
    def popularity_candidates(self, top_n: int = 20) -> List[int]:
        """
        Generate candidates based on popularity.
        
        Returns the most popular items sorted by popularity score.
        
        Args:
            top_n: Number of top popular items to return (default: 20).
            
        Returns:
            List of most popular item_ids, sorted by popularity.
            
        Example:
            >>> gen = CandidateGenerator(user_items, item_sim, item_pop)
            >>> gen.popularity_candidates(top_n=5)
            [10, 8, 5, 3, 1]
        """
        # Sort items by popularity score (descending)
        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N item ids
        return [item_id for item_id, score in sorted_items[:top_n]]
    
    def hybrid_candidates(self, user_id: int, top_n: int = 20) -> List[int]:
        """
        Generate candidates using a hybrid approach.
        
        Combines results from collaborative, content-based, and popularity
        methods. Removes duplicates and returns top items.
        
        Priority order: collaborative > content-based > popularity
        
        Args:
            user_id: The user to generate recommendations for.
            top_n: Maximum number of items to return (default: 20).
            
        Returns:
            List of recommended item_ids (max 20).
            
        Example:
            >>> gen = CandidateGenerator(user_items, item_sim, item_pop)
            >>> gen.hybrid_candidates(1)
            [5, 6, 7, 4, 8, 10, ...]
        """
        # Get user's history to filter out seen items
        user_history = self._get_user_history(user_id)
        
        # Collect candidates from all methods
        collaborative = self.collaborative_candidates(user_id)
        content_based = self.content_based_candidates(user_id)
        popularity = self.popularity_candidates(top_n=top_n)
        
        # Combine results maintaining priority order
        # Use a list to preserve order, track seen items with a set
        combined: List[int] = []
        seen: Set[int] = set()
        
        # Add collaborative candidates first (highest priority)
        for item_id in collaborative:
            if item_id not in seen and item_id not in user_history:
                combined.append(item_id)
                seen.add(item_id)
        
        # Add content-based candidates second
        for item_id in content_based:
            if item_id not in seen and item_id not in user_history:
                combined.append(item_id)
                seen.add(item_id)
        
        # Add popularity candidates last (for cold start and diversity)
        for item_id in popularity:
            if item_id not in seen and item_id not in user_history:
                combined.append(item_id)
                seen.add(item_id)
        
        # Return top N items
        return combined[:top_n]


if __name__ == "__main__":
    # Sample data for testing
    print("=" * 50)
    print("CandidateGenerator - Example Test Cases")
    print("=" * 50)
    
    # User-item interactions
    # User 1 liked items 1, 2, 3
    # User 2 liked items 2, 3, 4, 5
    # User 3 liked items 1, 5, 6
    # User 4 is a new user (cold start)
    user_items = {
        1: [1, 2, 3],
        2: [2, 3, 4, 5],
        3: [1, 5, 6],
        4: []  # New user - cold start
    }
    
    # Item similarities (content-based)
    # Item 1 is similar to items 7, 8
    # Item 2 is similar to items 8, 9
    # Item 3 is similar to items 9, 10
    item_similarities = {
        1: [7, 8],
        2: [8, 9],
        3: [9, 10],
        4: [10, 11],
        5: [11, 12],
        6: [12, 13]
    }
    
    # Item popularity scores
    item_popularity = {
        1: 100,
        2: 85,
        3: 70,
        4: 95,
        5: 60,
        6: 50,
        7: 40,
        8: 90,
        9: 30,
        10: 80,
        11: 20,
        12: 75,
        13: 10
    }
    
    # Create generator
    gen = CandidateGenerator(user_items, item_similarities, item_popularity)
    
    # Test collaborative filtering
    print("\n1. Collaborative Filtering Tests:")
    print("-" * 30)
    
    collab_1 = gen.collaborative_candidates(1)
    print(f"   User 1's history: {user_items[1]}")
    print(f"   Collaborative candidates: {collab_1}")
    print("   (Items from users who share items with User 1)")
    
    collab_4 = gen.collaborative_candidates(4)
    print(f"\n   User 4's history: {user_items[4]} (cold start)")
    print(f"   Collaborative candidates: {collab_4}")
    print("   (Empty - no history to find similar users)")
    
    # Test content-based filtering
    print("\n2. Content-Based Filtering Tests:")
    print("-" * 30)
    
    content_1 = gen.content_based_candidates(1)
    print(f"   User 1's history: {user_items[1]}")
    print(f"   Content-based candidates: {content_1}")
    print("   (Items similar to User 1's liked items)")
    
    content_4 = gen.content_based_candidates(4)
    print(f"\n   User 4's history: {user_items[4]} (cold start)")
    print(f"   Content-based candidates: {content_4}")
    print("   (Empty - no history to find similar items)")
    
    # Test popularity-based
    print("\n3. Popularity-Based Tests:")
    print("-" * 30)
    
    popular = gen.popularity_candidates(top_n=5)
    print(f"   Top 5 popular items: {popular}")
    print("   (Sorted by popularity score)")
    
    # Test hybrid approach
    print("\n4. Hybrid Approach Tests:")
    print("-" * 30)
    
    hybrid_1 = gen.hybrid_candidates(1, top_n=10)
    print(f"   User 1's history: {user_items[1]}")
    print(f"   Hybrid candidates (top 10): {hybrid_1}")
    print("   (Combined: collaborative + content + popularity)")
    
    hybrid_4 = gen.hybrid_candidates(4, top_n=10)
    print(f"\n   User 4's history: {user_items[4]} (cold start)")
    print(f"   Hybrid candidates (top 10): {hybrid_4}")
    print("   (Falls back to popularity for cold start users)")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
