import scripts.seed_data as seed_data
from data.database import SessionLocal
from data.models import Content, ContentSkill, Interaction, Skill, User, UserSkill


def test_seed_data_idempotent_and_minimum_counts() -> None:
    seed_data.create_tables()

    with SessionLocal() as db:
        users, users_inserted_first = seed_data.seed_users(db)
        content_items, content_inserted_first = seed_data.seed_content(db)
        skills, skills_inserted_first = seed_data.seed_skills(db)
        user_skills_inserted_first = seed_data.seed_user_skills(db, users, skills)
        content_skills_inserted_first = seed_data.seed_content_skills(db, content_items, skills)
        interactions_inserted_first = seed_data.seed_interactions(db, users, content_items)

        users_second, users_inserted_second = seed_data.seed_users(db)
        content_second, content_inserted_second = seed_data.seed_content(db)
        skills_second, skills_inserted_second = seed_data.seed_skills(db)
        user_skills_inserted_second = seed_data.seed_user_skills(db, users_second, skills_second)
        content_skills_inserted_second = seed_data.seed_content_skills(db, content_second, skills_second)
        interactions_inserted_second = seed_data.seed_interactions(db, users_second, content_second)

        assert users_inserted_second == 0
        assert content_inserted_second == 0
        assert skills_inserted_second == 0
        assert user_skills_inserted_second == 0
        assert content_skills_inserted_second == 0
        assert interactions_inserted_second == 0

        assert users_inserted_first >= 0
        assert content_inserted_first >= 0
        assert skills_inserted_first >= 0
        assert user_skills_inserted_first >= 0
        assert content_skills_inserted_first >= 0
        assert interactions_inserted_first >= 0

        total_users = db.query(User).count()
        total_content = db.query(Content).count()
        total_skills = db.query(Skill).count()
        total_user_skills = db.query(UserSkill).count()
        total_content_skills = db.query(ContentSkill).count()
        total_interactions = db.query(Interaction).count()

        assert total_users >= 10
        assert total_content >= 20
        assert total_skills >= 10
        assert total_user_skills > 0
        assert total_content_skills > 0
        assert total_interactions > 0
