from __future__ import annotations

import time
import uuid

from data.database import SessionLocal
from data.models import Interaction, User
from engine.orchestrator import RecommendationOrchestrator
import scripts.seed_data as seed_data


def test_orchestrator_recommendations_for_existing_user() -> None:
    seed_data.main()

    orchestrator = RecommendationOrchestrator()
    try:
        recommendations = orchestrator.get_recommendations(user_id=1, limit=5)
    finally:
        orchestrator.close()

    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5
    assert recommendations
    for item in recommendations:
        assert {"content_id", "score", "reason"}.issubset(item.keys())
        assert item["reason"] in {
            "Popular in your category",
            "Users like you liked this",
            "Similar to your previous interactions",
        }
        assert 0.0 <= item["score"] <= 1.0


def test_orchestrator_unknown_user_returns_empty_list() -> None:
    orchestrator = RecommendationOrchestrator()
    try:
        recommendations = orchestrator.get_recommendations(user_id=999_999, limit=5)
    finally:
        orchestrator.close()

    assert recommendations == []


def test_orchestrator_cold_start_and_cache_invalidation() -> None:
    seed_data.main()

    with SessionLocal() as db:
        user = User(name=f"cold-start-{uuid.uuid4().hex[:8]}", interests="Python, SQL")
        db.add(user)
        db.commit()
        db.refresh(user)
        user_id = user.id

    orchestrator = RecommendationOrchestrator()
    try:
        recs_first = orchestrator.get_recommendations(user_id=user_id, limit=3)
        recs_cached = orchestrator.get_recommendations(user_id=user_id, limit=3)

        assert recs_first
        assert recs_first is recs_cached
        assert all(item["reason"] == "Popular in your category" for item in recs_first)

        first_content_id = recs_first[0]["content_id"]
        orchestrator.record_feedback(
            user_id=user_id,
            content_id=first_content_id,
            interaction_type="view",
            rating=None,
        )

        recs_after_feedback = orchestrator.get_recommendations(user_id=user_id, limit=3)
        assert recs_after_feedback is not recs_cached
    finally:
        orchestrator.close()


def test_orchestrator_excludes_seen_and_duplicates() -> None:
    seed_data.main()

    with SessionLocal() as db:
        user = db.query(User).filter(User.name == "Aarav").first()
        assert user is not None
        seen_items = {
            row[0]
            for row in db.query(Interaction.content_id)
            .filter(Interaction.user_id == user.id)
            .all()
        }

    orchestrator = RecommendationOrchestrator()
    try:
        recommendations = orchestrator.get_recommendations(user_id=user.id, limit=10)
    finally:
        orchestrator.close()

    rec_ids = [item["content_id"] for item in recommendations]
    assert len(rec_ids) == len(set(rec_ids))
    assert set(rec_ids).isdisjoint(seen_items)


def test_orchestrator_cache_ttl_expiry() -> None:
    seed_data.main()

    orchestrator = RecommendationOrchestrator(cache_ttl_seconds=1)
    try:
        recs_first = orchestrator.get_recommendations(user_id=1, limit=5)
        recs_hit = orchestrator.get_recommendations(user_id=1, limit=5)
        assert recs_first is recs_hit

        time.sleep(1.2)

        recs_after_expiry = orchestrator.get_recommendations(user_id=1, limit=5)
        assert recs_after_expiry is not recs_hit
    finally:
        orchestrator.close()
