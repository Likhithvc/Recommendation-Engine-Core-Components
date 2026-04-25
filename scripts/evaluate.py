"""Evaluate recommendation quality using Precision@5, Recall@5, and NDCG@5.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from sqlalchemy.orm import joinedload

# Allow running this script directly from repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.database import SessionLocal
from data.models import Content, Interaction, User
from engine.candidate_gen import CandidateGenerator
from engine.evaluator import RecommendationEvaluator
from engine.scorer import RecommendationScorer
from engine.similarity import SimilarityCalculator

K = 5


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * w for w in widths)

    print(fmt_row(headers))
    print(separator)
    for row in rows:
        print(fmt_row(row))


def split_train_test(db) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """Split interactions into per-user train/test by created_at (70/30).

    Returns:
        training_history: user_id -> training item IDs (unique, ordered)
        test_items: user_id -> test item IDs (unique, ordered)
        ground_truth: user_id -> relevant test item IDs with rating >= 4
    """
    rows = (
        db.query(
            Interaction.user_id,
            Interaction.content_id,
            Interaction.rating,
            Interaction.created_at,
        )
        .order_by(Interaction.user_id.asc(), Interaction.created_at.asc(), Interaction.content_id.asc())
        .all()
    )

    per_user: dict[int, list[tuple[int, float | None]]] = {}
    for user_id, content_id, rating, _created_at in rows:
        per_user.setdefault(int(user_id), []).append((int(content_id), rating))

    training_history: dict[int, list[int]] = {}
    test_items: dict[int, list[int]] = {}
    ground_truth: dict[int, list[int]] = {}

    for user_id, interactions in per_user.items():
        if len(interactions) < 2:
            continue

        split_idx = max(1, int(len(interactions) * 0.7))
        split_idx = min(split_idx, len(interactions) - 1)

        train_chunk = interactions[:split_idx]
        test_chunk = interactions[split_idx:]

        train_seen: set[int] = set()
        train_ids: list[int] = []
        for content_id, _rating in train_chunk:
            if content_id not in train_seen:
                train_ids.append(content_id)
                train_seen.add(content_id)

        test_seen: set[int] = set()
        test_ids: list[int] = []
        gt_seen: set[int] = set()
        gt_ids: list[int] = []

        for content_id, rating in test_chunk:
            if content_id not in test_seen:
                test_ids.append(content_id)
                test_seen.add(content_id)

            if rating is not None and rating >= 4.0 and content_id not in gt_seen:
                gt_ids.append(content_id)
                gt_seen.add(content_id)

        training_history[user_id] = train_ids
        test_items[user_id] = test_ids
        ground_truth[user_id] = gt_ids

    return training_history, test_items, ground_truth


def build_item_similarities(contents: list[Content]) -> dict[int, list[int]]:
    """Build item similarity graph from skill overlap (Jaccard)."""
    similarity = SimilarityCalculator()
    item_skill_sets: dict[int, set[int]] = {
        int(content.id): {int(skill.id) for skill in content.skills} for content in contents
    }

    item_ids = list(item_skill_sets.keys())
    item_similarities: dict[int, list[int]] = {item_id: [] for item_id in item_ids}

    for idx, item_id in enumerate(item_ids):
        scored_neighbors: list[tuple[float, int]] = []
        for other_id in item_ids:
            if other_id == item_id:
                continue
            sim = similarity.jaccard_similarity(item_skill_sets[item_id], item_skill_sets[other_id])
            if sim > 0:
                scored_neighbors.append((sim, other_id))

        scored_neighbors.sort(key=lambda pair: pair[0], reverse=True)
        item_similarities[item_id] = [other_id for _sim, other_id in scored_neighbors[:10]]

    return item_similarities


def generate_recommendations(
    user_ids: list[int],
    training_history: dict[int, list[int]],
    contents: list[Content],
) -> dict[int, list[int]]:
    """Generate top-k recommendation IDs per user using training history only."""
    item_popularity: dict[int, float] = {int(content.id): float(content.popularity or 0) for content in contents}
    for item_ids in training_history.values():
        for item_id in item_ids:
            item_popularity[item_id] = item_popularity.get(item_id, 0.0) + 1.0

    item_similarities = build_item_similarities(contents)
    candidate_generator = CandidateGenerator(
        user_items=training_history,
        item_similarities=item_similarities,
        item_popularity=item_popularity,
    )

    max_popularity = max(item_popularity.values()) if item_popularity else 1.0
    max_item_id = max(item_popularity.keys()) if item_popularity else 1

    scorer = RecommendationScorer()

    def relevance_score(user_id: int, item_id: int, context: dict[str, Any]) -> float:
        history = set(training_history.get(user_id, []))
        if not history:
            return 0.4

        similar_items = set(item_similarities.get(item_id, []))
        if history.intersection(similar_items):
            return 0.9

        for seen in history:
            if item_id in item_similarities.get(seen, []):
                return 0.8
        return 0.3

    def popularity_score(_user_id: int, item_id: int, _context: dict[str, Any]) -> float:
        if max_popularity <= 0:
            return 0.0
        return min(1.0, item_popularity.get(item_id, 0.0) / max_popularity)

    def freshness_score(_user_id: int, item_id: int, _context: dict[str, Any]) -> float:
        return min(1.0, item_id / max_item_id)

    scorer.add_scorer("relevance", relevance_score, weight=2.0)
    scorer.add_scorer("popularity", popularity_score, weight=1.0)
    scorer.add_scorer("freshness", freshness_score, weight=0.5)

    recommendations: dict[int, list[int]] = {}
    for user_id in user_ids:
        candidates = candidate_generator.hybrid_candidates(user_id=user_id, top_n=max(K * 4, 20))
        seen_train = set(training_history.get(user_id, []))
        candidates = [item_id for item_id in candidates if item_id not in seen_train]

        ranked = scorer.rank_candidates(user_id=user_id, candidates=candidates, context={}, limit=K)
        recommendations[user_id] = [int(row["item_id"]) for row in ranked]

    return recommendations


def main() -> None:
    evaluator = RecommendationEvaluator()

    with SessionLocal() as db:
        user_ids = [row[0] for row in db.query(User.id).order_by(User.id.asc()).all()]
        if not user_ids:
            print("No users found in database. Seed data first using: python scripts/seed_data.py")
            return

        contents = db.query(Content).options(joinedload(Content.skills)).all()
        training_history, test_items, ground_truth = split_train_test(db)
        recommendations = generate_recommendations(user_ids, training_history, contents)

    # Evaluate only users with both non-empty recommendations and non-empty relevant items.
    eval_users = [
        uid
        for uid in user_ids
        if uid in training_history
        and len(training_history[uid]) > 0
        and uid in ground_truth
        and len(ground_truth[uid]) > 0
        and len(recommendations.get(uid, [])) > 0
    ]
    if not eval_users:
        print("No evaluable users found with both recommendations and relevant ground truth (rating >= 4).")
        return

    recommendations_eval = {uid: recommendations.get(uid, []) for uid in eval_users}
    ground_truth_eval = {uid: ground_truth[uid] for uid in eval_users}

    # Prefer a sample user with non-empty overlap for easier debugging.
    sample_user = eval_users[0]
    sample_train = training_history[sample_user]
    sample_test = test_items.get(sample_user, [])
    sample_recs = recommendations_eval[sample_user]
    sample_truth = ground_truth_eval[sample_user]
    sample_overlap = sorted(set(sample_recs).intersection(sample_truth))

    for uid in eval_users:
        candidate_train = training_history[uid]
        candidate_test = test_items.get(uid, [])
        candidate_recs = recommendations_eval[uid]
        candidate_truth = ground_truth_eval[uid]
        candidate_overlap = sorted(set(candidate_recs).intersection(candidate_truth))
        if candidate_overlap:
            sample_user = uid
            sample_train = candidate_train
            sample_test = candidate_test
            sample_recs = candidate_recs
            sample_truth = candidate_truth
            sample_overlap = candidate_overlap
            break

    print("\nDebug Sample")
    print(f"user_id: {sample_user}")
    print(f"training_items: {sample_train}")
    print(f"test_items: {sample_test}")
    print(f"recommendations@{K}: {sample_recs}")
    print(f"ground_truth_relevant: {sample_truth}")
    print(f"overlap: {sample_overlap}")
    if not sample_overlap:
        print("note: no overlap found for evaluated users in current dataset")

    aggregate = evaluator.evaluate_all(recommendations_eval, ground_truth_eval, k=K)

    print("\nRecommendation Evaluation Results")
    print(f"Users evaluated: {aggregate.get('num_users', 0)}")
    print(f"precision@{K}: {aggregate.get(f'precision@{K}', 0.0):.4f}")
    print(f"recall@{K}:    {aggregate.get(f'recall@{K}', 0.0):.4f}")
    print(f"ndcg@{K}:      {aggregate.get(f'ndcg@{K}', 0.0):.4f}")

    summary_rows = [
        [f"precision@{K}", f"{aggregate.get(f'precision@{K}', 0.0):.4f}"],
        [f"recall@{K}", f"{aggregate.get(f'recall@{K}', 0.0):.4f}"],
        [f"ndcg@{K}", f"{aggregate.get(f'ndcg@{K}', 0.0):.4f}"],
    ]

    print("\nSummary Table")
    _print_table(["Metric", "Value"], summary_rows)

    per_user_rows: list[list[str]] = []
    for user_id in eval_users:
        recs = recommendations_eval[user_id]
        relevant = set(ground_truth_eval[user_id])
        p_at_k = evaluator.precision_at_k(recs, relevant, K)
        r_at_k = evaluator.recall_at_k(recs, relevant, K)
        ndcg_at_k = evaluator.ndcg_at_k(recs, relevant, K)
        per_user_rows.append(
            [str(user_id), f"{p_at_k:.3f}", f"{r_at_k:.3f}", f"{ndcg_at_k:.3f}"]
        )

    print("\nPer-User Table")
    _print_table(["User ID", f"P@{K}", f"R@{K}", f"NDCG@{K}"], per_user_rows)


if __name__ == "__main__":
    main()
