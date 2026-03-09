# Mini Recommendation Engine

A lightweight, modular recommendation system built in Python. This project demonstrates core recommendation engine concepts including similarity calculations, candidate generation, scoring, and evaluation metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [How Recommendation Engines Work](#how-recommendation-engines-work)
- [Modules](#modules)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Example Output](#example-output)
- [License](#license)

---

## Project Overview

The Mini Recommendation Engine is an educational implementation of a recommendation system that covers the essential components found in production systems. It provides:

- **Similarity Calculations** — Measure how alike users or items are
- **Candidate Generation** — Find potential items to recommend
- **Scoring & Ranking** — Score and rank candidates by relevance
- **Evaluation Metrics** — Measure recommendation quality

This project is ideal for learning recommendation system fundamentals or as a foundation for building more complex systems.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   User Data  │    │  Item Data   │    │  Interaction │       │
│  │              │    │              │    │    History   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              SIMILARITY CALCULATOR                       │    │
│  │   • Cosine Similarity  • Jaccard  • Pearson Correlation │    │
│  └─────────────────────────────┬───────────────────────────┘    │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CANDIDATE GENERATOR                         │    │
│  │   • Collaborative Filtering  • Content-Based  • Popular │    │
│  └─────────────────────────────┬───────────────────────────┘    │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              RECOMMENDATION SCORER                       │    │
│  │   • Multiple Scoring Functions  • Weighted Combination  │    │
│  └─────────────────────────────┬───────────────────────────┘    │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              RANKED RECOMMENDATIONS                      │    │
│  │   [Item A: 0.95] [Item B: 0.87] [Item C: 0.72] ...      │    │
│  └─────────────────────────────┬───────────────────────────┘    │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EVALUATOR                                   │    │
│  │   • Precision@K  • Recall@K  • NDCG@K                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Recommendation Engines Work

Recommendation engines predict user preferences and suggest relevant items. Here's how the core concepts work:

### 1. Similarity Measurement

Before making recommendations, we need to understand relationships between users and items:

| Method | Description | Use Case |
|--------|-------------|----------|
| **Cosine Similarity** | Measures the angle between two vectors | Comparing user preference vectors |
| **Jaccard Similarity** | Overlap between two sets divided by union | Comparing categorical preferences |
| **Pearson Correlation** | Linear correlation between ratings | Finding users with similar rating patterns |

### 2. Candidate Generation

Generating a pool of potential recommendations:

- **Collaborative Filtering** — "Users like you also liked..."
- **Content-Based Filtering** — "Similar to items you've enjoyed..."
- **Popularity-Based** — "Trending items everyone loves..."
- **Hybrid Approach** — Combines all methods for better coverage

### 3. Scoring & Ranking

Candidates are scored using multiple factors:

```
Final Score = (relevance × w1) + (popularity × w2) + (recency × w3)
              ─────────────────────────────────────────────────────
                              (w1 + w2 + w3)
```

### 4. Evaluation

Measuring recommendation quality:

- **Precision@K** — How many recommended items are relevant?
- **Recall@K** — How many relevant items did we find?
- **NDCG@K** — Are relevant items ranked higher?

---

## Modules

### `similarity.py`

Calculates similarity metrics between users or items.

```python
from similarity import SimilarityCalculator

calc = SimilarityCalculator()

# Cosine similarity between user vectors
score = calc.cosine_similarity([1, 2, 3], [2, 3, 4])  # 0.9926

# Jaccard similarity between preference sets
score = calc.jaccard_similarity({'action', 'comedy'}, {'comedy', 'drama'})  # 0.33

# Pearson correlation between ratings
score = calc.pearson_correlation([5, 4, 3], [4, 3, 2])  # 1.0
```

**Methods:**
| Method | Parameters | Returns |
|--------|-----------|---------|
| `cosine_similarity(vec1, vec2)` | Two numeric lists | Float [0, 1] |
| `jaccard_similarity(set1, set2)` | Two Python sets | Float [0, 1] |
| `pearson_correlation(ratings1, ratings2)` | Two rating lists | Float [-1, 1] |

---

### `candidate_gen.py`

Generates recommendation candidates using multiple strategies.

```python
from candidate_gen import CandidateGenerator

gen = CandidateGenerator(user_items, item_similarities, item_popularity)

# Get candidates for a user
collab = gen.collaborative_candidates(user_id=1)
content = gen.content_based_candidates(user_id=1)
popular = gen.popularity_candidates(top_n=10)
hybrid = gen.hybrid_candidates(user_id=1, top_n=20)
```

**Methods:**
| Method | Description |
|--------|-------------|
| `collaborative_candidates(user_id)` | Items from similar users |
| `content_based_candidates(user_id)` | Items similar to user's history |
| `popularity_candidates(top_n)` | Most popular items |
| `hybrid_candidates(user_id, top_n)` | Combined approach |

---

### `scorer.py`

Scores and ranks candidates using configurable scoring functions.

```python
from scorer import RecommendationScorer

scorer = RecommendationScorer()

# Register scoring functions with weights
scorer.add_scorer("relevance", relevance_fn, weight=2.0)
scorer.add_scorer("popularity", popularity_fn, weight=1.0)

# Score a single item
score, explanation = scorer.calculate_score(user_id=1, item_id=42, context={})

# Rank multiple candidates
ranked = scorer.rank_candidates(user_id=1, candidates=[1, 2, 3], limit=10)
```

**Methods:**
| Method | Description |
|--------|-------------|
| `add_scorer(name, function, weight)` | Register a scoring function |
| `calculate_score(user_id, item_id, context)` | Get score with explanation |
| `rank_candidates(user_id, candidates, limit)` | Rank and return top items |

---

### `evaluator.py`

Evaluates recommendation quality using standard metrics.

```python
from evaluator import RecommendationEvaluator

evaluator = RecommendationEvaluator()

recommendations = [1, 2, 3, 4, 5]
relevant_items = {1, 3, 5, 7}

# Individual metrics
precision = evaluator.precision_at_k(recommendations, relevant_items, k=3)
recall = evaluator.recall_at_k(recommendations, relevant_items, k=3)
ndcg = evaluator.ndcg_at_k(recommendations, relevant_items, k=3)

# Evaluate across all users
metrics = evaluator.evaluate_all(recommendations_dict, ground_truth_dict, k=5)
```

**Methods:**
| Method | Formula | Returns |
|--------|---------|---------|
| `precision_at_k` | relevant_in_top_k / k | Float [0, 1] |
| `recall_at_k` | relevant_found / total_relevant | Float [0, 1] |
| `ndcg_at_k` | DCG / IDCG | Float [0, 1] |
| `evaluate_all` | Average metrics across users | Dict |

---

### `recommender_engine.py`

**The orchestrator module** that integrates all components into a unified recommendation pipeline.

This module combines:
- **SimilarityCalculator** — For measuring user/item similarities
- **CandidateGenerator** — For generating recommendation candidates  
- **RecommendationScorer** — For scoring and ranking candidates

```
┌─────────────────────────────────────────────────────────────┐
│                  RecommenderEngine Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Request                                               │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────────┐                                   │
│   │ Candidate Generator │  → Hybrid candidates              │
│   └──────────┬──────────┘                                   │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │ Recommendation      │  → Score each candidate           │
│   │ Scorer              │    (relevance + popularity +      │
│   └──────────┬──────────┘     recency)                      │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │ Rank & Return       │  → Top N recommendations          │
│   └─────────────────────┘                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Quick Start:**

```python
from recommender_engine import RecommenderEngine

# Prepare your data
user_items = {
    1: [1, 2, 3],      # User 1 liked items 1, 2, 3
    2: [2, 3, 4, 5],   # User 2 liked items 2, 3, 4, 5
    3: []              # User 3 is new (cold start)
}

item_similarities = {
    1: [2, 7],         # Item 1 is similar to items 2, 7
    2: [1, 3],
    3: [2, 4],
    # ... more items
}

item_popularity = {
    1: 95, 2: 85, 3: 70, 4: 90, 5: 60,
    6: 50, 7: 40, 8: 88, 9: 30, 10: 75
}

# Initialize the engine
engine = RecommenderEngine(user_items, item_similarities, item_popularity)

# Generate recommendations
recommendations = engine.recommend(user_id=1, limit=5)

# Print results
for rec in recommendations:
    print(f"Item {rec['item_id']}: Score {rec['score']:.2f}")
```

**Output:**
```
Item 4: Score 0.71
Item 7: Score 0.60
Item 8: Score 0.55
Item 10: Score 0.54
Item 5: Score 0.42
```

**Adding Custom Scorers:**

```python
# Define a custom scoring function
def seasonal_boost(user_id, item_id, context):
    """Boost items that are seasonal/promotional."""
    seasonal_items = {4, 8, 10}  # Items on promotion
    return 1.0 if item_id in seasonal_items else 0.0

# Add to engine with a weight
engine.add_custom_scorer("seasonal", seasonal_boost, weight=1.5)

# Get updated recommendations
recommendations = engine.recommend(user_id=1, limit=5)
```

**Methods:**
| Method | Description |
|--------|-------------|
| `recommend(user_id, limit)` | Full pipeline: generate → score → rank → return |
| `add_custom_scorer(name, fn, weight)` | Add domain-specific scoring logic |
| `get_user_history(user_id)` | Get items a user has interacted with |
| `get_similar_items(item_id)` | Get items similar to a given item |

**Default Scorers:**
| Scorer | Weight | Description |
|--------|--------|-------------|
| `relevance` | 2.0 | How relevant is the item to user's history |
| `popularity` | 1.0 | How popular is the item overall |
| `recency` | 0.5 | How new/fresh is the item |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/mini-recommendation-engine.git
cd mini-recommendation-engine
```

2. **Create a virtual environment** (recommended)

```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# macOS/Linux
python -m venv myenv
source myenv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running Tests

Run the comprehensive test suite:

```bash
python test.py
```

Test individual modules:

```bash
python similarity.py
python candidate_gen.py
python scorer.py
python evaluator.py
```

---

## Example Output

Running `python test.py` produces:

```
############################################################
#  RECOMMENDATION ENGINE - CORE COMPONENTS TEST SUITE
############################################################

============================================================
 1. Testing SimilarityCalculator
============================================================

  Cosine Similarity
  ----------------------------------------
    User 1 vs User 2: 0.9912
    User 1 vs User 3: 0.6364
    ✓ Cosine similarity tests PASSED

  Jaccard Similarity
  ----------------------------------------
    User 1 preferences: {'comedy', 'sci-fi', 'action'}
    User 2 preferences: {'drama', 'sci-fi', 'action'}
    Jaccard similarity: 0.5000
    ✓ Jaccard similarity tests PASSED

============================================================
 2. Testing CandidateGenerator
============================================================

  Hybrid Candidates
  ----------------------------------------
    User 1 hybrid candidates: [4, 5, 6, 7, 8, 10, 9]
    User 5 hybrid (cold start): [1, 4, 8, 2, 10]
    ✓ Hybrid candidates tests PASSED

============================================================
 3. Testing RecommendationScorer
============================================================

  Rank Candidates
  ----------------------------------------
    Top 5 Ranked:
      #1: Item 4 - Score: 0.7857
      #2: Item 5 - Score: 0.6429
      #3: Item 8 - Score: 0.6000
    ✓ Ranking tests PASSED

============================================================
 4. Testing RecommendationEvaluator
============================================================

  Evaluate All Users
  ----------------------------------------
    Average Metrics (K=5):
      Precision@5: 0.5600
      Recall@5: 0.9333
      NDCG@5: 0.8931
    ✓ Evaluate all tests PASSED

============================================================
 TEST SUMMARY
============================================================
    SimilarityCalculator: ✓ PASSED
    CandidateGenerator: ✓ PASSED
    RecommendationScorer: ✓ PASSED
    RecommendationEvaluator: ✓ PASSED
    Integration: ✓ PASSED

    Total: 5/5 test suites passed

    🎉 ALL TESTS PASSED! 🎉
```

---

## Project Structure

```
mini-recommendation-engine/
├── recommender_engine.py  # Main engine (orchestrates full pipeline)
├── similarity.py          # Similarity calculations
├── candidate_gen.py       # Candidate generation strategies
├── scorer.py              # Scoring and ranking
├── evaluator.py           # Evaluation metrics
├── test.py                # Comprehensive test suite
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by production recommendation systems at Netflix, Spotify, and Amazon
- Built for educational purposes to demonstrate core recommendation concepts

---

<p align="center">
  Made with ❤️ for the recommendation systems community
</p>
