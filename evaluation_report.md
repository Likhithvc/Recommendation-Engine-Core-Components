# Recommendation System Evaluation Report

## 1. Dataset and Experimental Setup

This recommendation system was evaluated on a structured interaction dataset designed to reflect category-level user preferences.

- Number of users: **10**
- Number of content items: **21**
- Content domains: **AI**, **Web Dev**, **Data Science**
- Interaction signal: explicit ratings with user-content interactions

The dataset is intentionally small but semantically organized to support controlled offline evaluation.

## 2. Evaluation Methodology

An offline temporal evaluation protocol was used to simulate practical recommendation behavior.

- **Train-test split**: For each user, interactions were ordered by `created_at` and split into **70% train** and **30% test**.
- **Ground truth relevance**: In the test partition, items with **rating >= 4** were treated as relevant.
- **Recommendation rule**: Recommended items excluded all user-seen items from the training history.
- **Evaluation scope**: Users with no relevant items in the test split were skipped to avoid distorted recall and NDCG values.

This protocol reduces leakage and better reflects the task of predicting future user preferences from past behavior.

## 3. Metrics

The system was measured with top-k ranking metrics at $k=5$:

- **Precision@5**: Fraction of top-5 recommendations that are relevant.
- **Recall@5**: Fraction of relevant test items retrieved in top-5 recommendations.
- **NDCG@5**: Ranking-sensitive metric that rewards placing relevant items higher in the list.

## 4. Results

- **Precision@5: 0.24**
- **Recall@5: 0.53**
- **NDCG@5: 0.46**

## 5. Interpretation of Results

### Precision@5 (0.24)
Precision is moderate, meaning that roughly 1.2 out of the top 5 recommended items are relevant on average. This is expected in a hybrid recommender where diversity and coverage are preserved alongside strict personalization.

### Recall@5 (0.53)
Recall is comparatively strong, indicating that the system recovers more than half of relevant items for evaluable users. This suggests effective candidate coverage and reasonable user-interest matching.

### NDCG@5 (0.46)
NDCG indicates reasonable ranking quality. Relevant items are often present, but they are not always placed at the very top positions, leaving room for ranking calibration improvements.

## 6. Per-User Variation Analysis

Performance varies substantially across users. Some users achieve non-zero precision/recall, while others score zero across all metrics. This variation is primarily driven by:

- uneven interaction density per user,
- limited relevant items in the test split for specific users,
- narrow historical preference signals for some profiles.

Such per-user variance is common in small recommendation datasets and highlights the need for richer behavior signals.

## 7. Challenges

- **Small dataset size**: With only 10 users and 21 items, estimates are sensitive to minor interaction changes.
- **Sparse interactions**: Some users have insufficient historical depth for stable personalization.

## 8. Trade-Offs: Personalization vs Popularity

The system balances personalized relevance signals with popularity priors.

- Increasing personalization weight can improve ranking alignment for active users.
- Popularity helps stability and cold-start behavior but may reduce user-specific precision if overemphasized.

The current results reflect this trade-off: healthy recall and moderate precision.

## 9. Future Improvements

- **Matrix factorization** for latent user-item preference modeling.
- **Embedding-based representations** for richer semantic similarity.
- **Larger dataset collection** to improve statistical reliability and model generalization.

## 10. Conclusion

The system provides meaningful personalized recommendations under a realistic offline protocol. Results show good coverage (recall), moderate precision, and reasonable ranking quality, making the current pipeline a solid baseline for future model-driven improvements.
