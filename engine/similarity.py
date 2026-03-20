"""
Similarity Calculator Module

This module provides a SimilarityCalculator class with methods
to compute various similarity and correlation metrics.
"""

import math
from typing import List, Set


class SimilarityCalculator:
    """
    A class to calculate various similarity and correlation metrics.
    
    Methods:
        cosine_similarity: Calculate cosine similarity between two vectors.
        jaccard_similarity: Calculate Jaccard similarity between two sets.
        pearson_correlation: Calculate Pearson correlation coefficient.
    """
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate the cosine similarity between two numeric vectors.
        
        Cosine similarity measures the cosine of the angle between two vectors.
        A value of 1 means vectors point in the same direction, 0 means orthogonal.
        
        Args:
            vec1: First numeric list (vector).
            vec2: Second numeric list (vector).
            
        Returns:
            A float between 0 and 1 representing similarity.
            Returns 0 for zero vectors or mismatched lengths.
            
        Example:
            >>> calc = SimilarityCalculator()
            >>> calc.cosine_similarity([1, 2, 3], [1, 2, 3])
            1.0
        """
        # Check for valid inputs
        if len(vec1) != len(vec2):
            return 0.0
        
        if len(vec1) == 0:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Handle zero vectors
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Clamp to [0, 1] range (handle floating point errors)
        # Note: True cosine similarity ranges from -1 to 1, but for 
        # recommendation systems we often use 0 to 1
        return max(0.0, min(1.0, similarity))
    
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """
        Calculate the Jaccard similarity between two sets.
        
        Jaccard similarity is the size of intersection divided by 
        the size of union of two sets.
        
        Args:
            set1: First Python set.
            set2: Second Python set.
            
        Returns:
            A float between 0 and 1 representing similarity.
            Returns 0 if both sets are empty.
            
        Example:
            >>> calc = SimilarityCalculator()
            >>> calc.jaccard_similarity({1, 2, 3}, {2, 3, 4})
            0.5
        """
        # Handle empty sets
        if len(set1) == 0 and len(set2) == 0:
            return 0.0
        
        # Calculate intersection and union
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # Return Jaccard index
        return len(intersection) / len(union)
    
    def pearson_correlation(self, ratings1: List[float], ratings2: List[float]) -> float:
        """
        Calculate the Pearson correlation coefficient between two rating lists.
        
        Pearson correlation measures the linear relationship between two variables.
        Values range from -1 (perfect negative) to 1 (perfect positive).
        For recommendation systems, we often normalize to 0-1 range.
        
        Args:
            ratings1: First list of ratings.
            ratings2: Second list of ratings.
            
        Returns:
            A float between -1 and 1 representing correlation.
            Returns 0 if inputs are invalid or have zero variance.
            
        Example:
            >>> calc = SimilarityCalculator()
            >>> calc.pearson_correlation([1, 2, 3], [1, 2, 3])
            1.0
        """
        # Check for valid inputs
        if len(ratings1) != len(ratings2):
            return 0.0
        
        n = len(ratings1)
        if n == 0:
            return 0.0
        
        # Calculate means
        mean1 = sum(ratings1) / n
        mean2 = sum(ratings2) / n
        
        # Calculate deviations from mean
        dev1 = [x - mean1 for x in ratings1]
        dev2 = [y - mean2 for y in ratings2]
        
        # Calculate sum of products of deviations
        sum_product = sum(d1 * d2 for d1, d2 in zip(dev1, dev2))
        
        # Calculate sum of squared deviations
        sum_sq1 = sum(d * d for d in dev1)
        sum_sq2 = sum(d * d for d in dev2)
        
        # Handle zero variance (all values are the same)
        if sum_sq1 == 0 or sum_sq2 == 0:
            return 0.0
        
        # Calculate Pearson correlation
        correlation = sum_product / (math.sqrt(sum_sq1) * math.sqrt(sum_sq2))
        
        # Clamp to [-1, 1] range (handle floating point errors)
        return max(-1.0, min(1.0, correlation))


if __name__ == "__main__":
    # Example test cases
    print("=" * 50)
    print("SimilarityCalculator - Example Test Cases")
    print("=" * 50)
    
    calc = SimilarityCalculator()
    
    # Test Cosine Similarity
    print("\n1. Cosine Similarity Tests:")
    print("-" * 30)
    
    # Identical vectors
    vec_a = [1, 2, 3]
    vec_b = [1, 2, 3]
    result = calc.cosine_similarity(vec_a, vec_b)
    print(f"   Identical vectors {vec_a} and {vec_b}")
    print(f"   Result: {result:.4f} (expected: 1.0)")
    
    # Orthogonal vectors
    vec_c = [1, 0]
    vec_d = [0, 1]
    result = calc.cosine_similarity(vec_c, vec_d)
    print(f"\n   Orthogonal vectors {vec_c} and {vec_d}")
    print(f"   Result: {result:.4f} (expected: 0.0)")
    
    # Zero vector
    vec_e = [0, 0, 0]
    vec_f = [1, 2, 3]
    result = calc.cosine_similarity(vec_e, vec_f)
    print(f"\n   Zero vector {vec_e} and {vec_f}")
    print(f"   Result: {result:.4f} (expected: 0.0)")
    
    # Test Jaccard Similarity
    print("\n2. Jaccard Similarity Tests:")
    print("-" * 30)
    
    # Overlapping sets
    set_a = {1, 2, 3}
    set_b = {2, 3, 4}
    result = calc.jaccard_similarity(set_a, set_b)
    print(f"   Sets {set_a} and {set_b}")
    print(f"   Intersection: {set_a & set_b}, Union: {set_a | set_b}")
    print(f"   Result: {result:.4f} (expected: 0.5)")
    
    # Identical sets
    set_c = {1, 2, 3}
    set_d = {1, 2, 3}
    result = calc.jaccard_similarity(set_c, set_d)
    print(f"\n   Identical sets {set_c} and {set_d}")
    print(f"   Result: {result:.4f} (expected: 1.0)")
    
    # Empty sets
    set_e = set()
    set_f = set()
    result = calc.jaccard_similarity(set_e, set_f)
    print(f"\n   Empty sets {set_e} and {set_f}")
    print(f"   Result: {result:.4f} (expected: 0.0)")
    
    # Test Pearson Correlation
    print("\n3. Pearson Correlation Tests:")
    print("-" * 30)
    
    # Perfect positive correlation
    ratings_a = [1, 2, 3, 4, 5]
    ratings_b = [2, 4, 6, 8, 10]
    result = calc.pearson_correlation(ratings_a, ratings_b)
    print(f"   Ratings {ratings_a} and {ratings_b}")
    print(f"   Result: {result:.4f} (expected: 1.0)")
    
    # Perfect negative correlation
    ratings_c = [1, 2, 3, 4, 5]
    ratings_d = [5, 4, 3, 2, 1]
    result = calc.pearson_correlation(ratings_c, ratings_d)
    print(f"\n   Ratings {ratings_c} and {ratings_d}")
    print(f"   Result: {result:.4f} (expected: -1.0)")
    
    # No correlation (constant values)
    ratings_e = [3, 3, 3, 3, 3]
    ratings_f = [1, 2, 3, 4, 5]
    result = calc.pearson_correlation(ratings_e, ratings_f)
    print(f"\n   Constant ratings {ratings_e} and {ratings_f}")
    print(f"   Result: {result:.4f} (expected: 0.0)")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
