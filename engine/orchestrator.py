"""Recommendation orchestration layer built on existing engine modules."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from data.database import SessionLocal
from data.models import Interaction, User
from data.repositories import (
    ContentRepository,
    InteractionRepository,
    SkillRepository,
    UserRepository,
)
from engine.candidate_gen import CandidateGenerator
from engine.scorer import RecommendationScorer
from engine.similarity import SimilarityCalculator


@dataclass(frozen=True)
class ScoringWeights:
    # Fixed ranking profile for precision-oriented ordering.
    relevance: float = 0.7
    popularity: float = 0.2
    recency: float = 0.1


class RecommendationOrchestrator:
    """Coordinates data loading, candidate generation, scoring, and feedback."""

    def __init__(
        self,
        db: Session | None = None,
        weights: ScoringWeights | None = None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        self._external_db = db is not None
        self.db = db or SessionLocal()

        self.user_repo = UserRepository(self.db)
        self.content_repo = ContentRepository(self.db)
        self.interaction_repo = InteractionRepository(self.db)
        self.skill_repo = SkillRepository(self.db)

        self.similarity = SimilarityCalculator()
        self.weights = weights or ScoringWeights()
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))

        self._cache: dict[int, dict[str, Any]] = {}

    def close(self) -> None:
        if not self._external_db:
            self.db.close()

    def _load_structures(self) -> dict[str, Any]:
        all_content = self.content_repo.get_all_content()

        user_items: dict[int, list[int]] = {}
        user_recent_items: dict[int, list[int]] = {}
        item_popularity: dict[int, float] = {c.id: float(c.popularity or 0) for c in all_content}
        item_categories: dict[int, str] = {c.id: c.category for c in all_content}
        category_popularity: dict[str, dict[int, float]] = {}
        content_by_id = {c.id: c for c in all_content}

        interactions = self.db.query(Interaction).all()
        for interaction in interactions:
            user_items.setdefault(interaction.user_id, []).append(interaction.content_id)
            item_popularity[interaction.content_id] = item_popularity.get(interaction.content_id, 0.0) + 1.0

            content = content_by_id.get(interaction.content_id)
            if content is not None:
                category_scores = category_popularity.setdefault(content.category, {})
                category_scores[interaction.content_id] = category_scores.get(interaction.content_id, 0.0) + 1.0

        for user_id, history in user_items.items():
            user_recent_items[user_id] = list(reversed(history[-5:]))

        user_preferred_category: dict[int, str] = {}
        for user_id, history in user_items.items():
            category_counts: dict[str, int] = {}
            for item_id in history:
                category = item_categories.get(item_id)
                if not category:
                    continue
                category_counts[category] = category_counts.get(category, 0) + 1

            if category_counts:
                preferred = max(category_counts.items(), key=lambda pair: pair[1])[0]
                user_preferred_category[user_id] = preferred

        for user in self.db.query(User).all():
            user_id = user.id
            if user_id in user_preferred_category:
                continue
            interests = (user.interests or "").strip()
            if not interests:
                continue
            user_preferred_category[user_id] = interests.split(",", 1)[0].strip()

        item_skill_sets: dict[int, set[int]] = {
            c.id: {skill.id for skill in c.skills} for c in all_content
        }

        item_similarities: dict[int, list[int]] = {c.id: [] for c in all_content}
        content_ids = list(item_skill_sets.keys())
        for idx, content_id in enumerate(content_ids):
            base_skills = item_skill_sets.get(content_id, set())
            scored_neighbors: list[tuple[float, int]] = []

            for other_id in content_ids[idx + 1 :]:
                other_skills = item_skill_sets.get(other_id, set())
                sim = self.similarity.jaccard_similarity(base_skills, other_skills)
                if sim > 0:
                    scored_neighbors.append((sim, other_id))
                    item_similarities.setdefault(other_id, []).append(content_id)

            scored_neighbors.sort(key=lambda pair: pair[0], reverse=True)
            item_similarities[content_id].extend([neighbor for _, neighbor in scored_neighbors[:10]])

        max_popularity = max(item_popularity.values()) if item_popularity else 1.0
        return {
            "all_content": all_content,
            "content_by_id": content_by_id,
            "user_items": user_items,
            "user_recent_items": user_recent_items,
            "user_preferred_category": user_preferred_category,
            "item_popularity": item_popularity,
            "item_similarities": item_similarities,
            "item_categories": item_categories,
            "category_popularity": category_popularity,
            "max_popularity": max_popularity,
        }

    def _build_scorer(self, structures: dict[str, Any]) -> RecommendationScorer:
        scorer = RecommendationScorer()

        user_items = structures["user_items"]
        user_recent_items = structures["user_recent_items"]
        user_preferred_category = structures["user_preferred_category"]
        item_categories = structures["item_categories"]
        item_similarities = structures["item_similarities"]
        item_popularity = structures["item_popularity"]
        max_popularity = structures["max_popularity"]

        def relevance_score(user_id: int, item_id: int, context: dict[str, Any]) -> float:
            return self._relevance_signal(user_id, item_id, structures)

        def popularity_score(user_id: int, item_id: int, context: dict[str, Any]) -> float:
            popularity = item_popularity.get(item_id, 0.0)
            if max_popularity <= 0:
                return 0.0

            # Damp popularity so it supports ranking without overpowering personalization.
            normalized = popularity / max_popularity
            return min(1.0, normalized ** 0.5)

        def recency_score(user_id: int, item_id: int, context: dict[str, Any]) -> float:
            created_at = context["content_by_id"][item_id].id
            max_id = max(context["content_by_id"].keys()) if context["content_by_id"] else 1
            return min(1.0, created_at / max_id)

        scorer.add_scorer("relevance", relevance_score, weight=self.weights.relevance)
        scorer.add_scorer("popularity", popularity_score, weight=self.weights.popularity)
        scorer.add_scorer("recency", recency_score, weight=self.weights.recency)

        return scorer

    def _relevance_signal(self, user_id: int, item_id: int, structures: dict[str, Any]) -> float:
        """Compute relevance with a strong dominant-category boost."""
        user_items = structures["user_items"]
        user_recent_items = structures["user_recent_items"]
        user_preferred_category = structures["user_preferred_category"]
        item_categories = structures["item_categories"]
        item_similarities = structures["item_similarities"]

        history = set(user_items.get(user_id, []))
        preferred_category = user_preferred_category.get(user_id)
        item_category = item_categories.get(item_id)
        category_match = bool(preferred_category and item_category == preferred_category)

        if not history:
            base = 0.52
            if category_match:
                base += 0.20
            return min(1.0, base)

        recent_items = user_recent_items.get(user_id, [])[:3]
        recency_hit = any(item_id in item_similarities.get(seen_id, []) for seen_id in recent_items)

        similar_to_candidate = set(item_similarities.get(item_id, []))
        if history.intersection(similar_to_candidate):
            base = 0.86
            if recency_hit:
                base += 0.05
            if category_match:
                base += 0.10
            return min(1.0, base)

        for seen in history:
            if item_id in item_similarities.get(seen, []):
                base = 0.73
                if recency_hit:
                    base += 0.06
                if category_match:
                    base += 0.13
                return min(1.0, base)

        if category_match:
            return 0.60

        return 0.14

    def _candidate_confidence(self, user_id: int, item_id: int, structures: dict[str, Any]) -> float:
        """Estimate candidate confidence before ranking to trim noisy items."""
        history = set(structures["user_items"].get(user_id, []))
        preferred_category = structures["user_preferred_category"].get(user_id)
        item_category = structures["item_categories"].get(item_id)

        score = 0.0
        if preferred_category and item_category == preferred_category:
            score += 0.4

        if history:
            for seen in history:
                if item_id in structures["item_similarities"].get(seen, []):
                    score += 0.45
                    break

        popularity = structures["item_popularity"].get(item_id, 0.0)
        max_popularity = structures["max_popularity"]
        if max_popularity > 0:
            score += min(0.2, (popularity / max_popularity) * 0.2)

        return min(1.0, score)

    def _reason(self, user_id: int, item_id: int, structures: dict[str, Any]) -> str:
        user_history = set(structures["user_items"].get(user_id, []))
        preferred_category = structures["user_preferred_category"].get(user_id)
        item_category = structures["item_categories"].get(item_id)

        if not user_history:
            if preferred_category and item_category == preferred_category:
                return "Popular in your category"
            return "Popular in your category"

        collaborative_users = 0
        for other_user_id, other_items in structures["user_items"].items():
            if other_user_id == user_id:
                continue
            if user_history.intersection(set(other_items)) and item_id in other_items:
                collaborative_users += 1

        if collaborative_users > 0:
            return "Users like you liked this"

        similar_to_history = False
        for seen_item in user_history:
            if item_id in structures["item_similarities"].get(seen_item, []):
                similar_to_history = True
                break
            if seen_item in structures["item_similarities"].get(item_id, []):
                similar_to_history = True
                break

        if similar_to_history:
            return "Similar to your previous interactions"

        if preferred_category and item_category == preferred_category:
            return "Popular in your category"

        return "Similar to your previous interactions"

    def get_recommendations(self, user_id: int, limit: int = 5) -> list[dict[str, Any]]:
        now = time.time()
        cached = self._cache.get(user_id)
        if cached is not None and cached["expires_at"] > now:
            cached_limit = int(cached.get("limit", 0))
            cached_recommendations: list[dict[str, Any]] = cached["recommendations"]
            if limit <= cached_limit:
                print("Cache HIT")
                if limit == cached_limit:
                    return cached_recommendations
                return cached_recommendations[:limit]

        print("Cache MISS")

        user = self.user_repo.get_user(user_id)
        if user is None:
            return []

        structures = self._load_structures()
        candidate_generator = CandidateGenerator(
            user_items=structures["user_items"],
            item_similarities=structures["item_similarities"],
            item_popularity=structures["item_popularity"],
            user_recent_items=structures["user_recent_items"],
            item_categories=structures["item_categories"],
            user_preferred_category=structures["user_preferred_category"],
            category_popularity=structures["category_popularity"],
        )

        seen_items = set(structures["user_items"].get(user_id, []))

        if seen_items:
            candidate_ids = candidate_generator.hybrid_candidates(user_id, top_n=max(limit * 4, 20))
        else:
            # Cold-start users: popularity within known interest category, else global popularity.
            candidate_ids = candidate_generator.popularity_candidates(top_n=max(limit * 2, 20), user_id=user_id)

        candidate_ids = [item_id for item_id in candidate_ids if item_id not in seen_items]
        candidate_ids = list(dict.fromkeys(candidate_ids))

        # Filter weak-relevance candidates to improve precision/NDCG without expanding pool size.
        if len(candidate_ids) > limit:
            filtered = [
                item_id
                for item_id in candidate_ids
                if self._relevance_signal(user_id, item_id, structures) >= 0.20
            ]
            if len(filtered) >= limit:
                candidate_ids = filtered

        if not candidate_ids:
            self._cache[user_id] = {
                "recommendations": [],
                "limit": limit,
                "expires_at": now + self.cache_ttl_seconds,
            }
            return []

        scorer = self._build_scorer(structures)
        content_by_id = structures["content_by_id"]

        ranked = scorer.rank_candidates(
            user_id=user_id,
            candidates=candidate_ids,
            context={
                "content_by_id": content_by_id,
                "user_history_length": len(seen_items),
                "fixed_weight_profile": "precision_v1",
            },
            limit=limit,
        )

        recommendations: list[dict[str, Any]] = []
        for row in ranked:
            content_id = row["item_id"]
            content = content_by_id.get(content_id)
            if content is None:
                continue

            reason = self._reason(user_id, content.id, structures)

            recommendations.append(
                {
                    "content_id": content.id,
                    "title": content.title,
                    "score": round(row["score"], 4),
                    "reason": reason,
                    # Backward-compatible field for existing consumers.
                    "explanation": reason,
                }
            )

        self._cache[user_id] = {
            "recommendations": recommendations,
            "limit": limit,
            "expires_at": now + self.cache_ttl_seconds,
        }
        return recommendations

    def record_feedback(
        self,
        user_id: int,
        content_id: int,
        interaction_type: str,
        rating: float | None = None,
    ) -> None:
        self.interaction_repo.record_interaction(
            user_id=user_id,
            content_id=content_id,
            type=interaction_type,
            rating=rating,
        )

        if user_id in self._cache:
            del self._cache[user_id]
