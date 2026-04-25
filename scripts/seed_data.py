"""Seed script for local SQLite database with strong collaborative patterns.

Usage:
    python scripts/seed_data.py

This script generates:
- 30 users (10 per category)
- 60-80 content items (72 total: 24 per category)
- 10-15 interactions per user
- Strong shared item overlap for collaborative filtering
- 10-15% cross-category noise
- Fixed random seed for reproducibility
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import func

# Allow running the script directly from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.database import Base, SessionLocal, engine
from data.models import Content, ContentSkill, Interaction, Skill, User, UserSkill

# Fixed seed for reproducibility
RNG = random.Random(42)

CATEGORIES = ("AI", "Web Dev", "Data Science")
USERS_PER_CATEGORY = 10
CONTENT_ITEMS_PER_CATEGORY = 24
CORE_POOL_SIZE = 12
MANDATORY_SHARED_COUNT = 6

# 30 users: 10 AI, 10 Web Dev, 10 Data Science
USER_PROFILES = {
    "AI": [
        "Aarav", "Diya", "Kabir", "Arjun", "Vikram",
        "Amit", "Deepak", "Aditya", "Ravi", "Nikhil",
    ],
    "Web Dev": [
        "Meera", "Rohan", "Anaya", "Vivaan", "Ishaan",
        "Rajesh", "Sanjay", "Harpreet", "Suresh", "Rahul",
    ],
    "Data Science": [
        "Sara", "Zara", "Isha", "Pooja", "Sneha",
        "Anjali", "Divya", "Sakshi", "Neha", "Simran",
    ],
}

# Flatten into list for backward compatibility
USER_CATEGORY_MAP = {}
for category, names in USER_PROFILES.items():
    for name in names:
        USER_CATEGORY_MAP[name] = category

# Category-specific skills
CATEGORY_SKILLS = {
    "AI": [
        "Python", "Machine Learning", "Deep Learning", "NLP", "MLOps",
        "TensorFlow", "PyTorch", "Computer Vision", "Neural Networks",
    ],
    "Web Dev": [
        "JavaScript", "HTML/CSS", "React", "API Design", "Backend",
        "TypeScript", "Node.js", "Vue.js", "CSS Grid",
    ],
    "Data Science": [
        "Python", "Statistics", "Pandas", "SQL", "Data Visualization",
        "Matplotlib", "NumPy", "Scikit-learn", "Data Cleaning",
    ],
}

# All unique skills (20-30 depending on overlaps)
SKILL_NAMES = sorted({skill for skills in CATEGORY_SKILLS.values() for skill in skills})

# Comprehensive content library: 100 items (34 AI, 33 Web Dev, 33 Data Science)
CONTENT_TEMPLATES = {
    "AI": [
        ("Intro to Machine Learning", "Beginner", 85),
        ("Neural Networks Fundamentals", "Intermediate", 88),
        ("Prompt Engineering Basics", "Beginner", 92),
        ("Transformers in Practice", "Advanced", 82),
        ("Computer Vision Essentials", "Intermediate", 79),
        ("MLOps for Production", "Advanced", 76),
        ("NLP with Python", "Intermediate", 81),
        ("Deep Learning Crash Course", "Intermediate", 85),
        ("CNN for Image Classification", "Advanced", 78),
        ("Reinforcement Learning Basics", "Advanced", 72),
        ("Transfer Learning Techniques", "Intermediate", 80),
        ("Time Series with LSTM", "Advanced", 74),
        ("Model Evaluation Methods", "Intermediate", 79),
        ("Feature Engineering for ML", "Intermediate", 83),
        ("Gradient Descent Deep Dive", "Advanced", 71),
        ("Bayesian Methods in ML", "Advanced", 70),
        ("Attention Mechanisms Explained", "Advanced", 75),
        ("Vision Transformers", "Advanced", 68),
        ("Text Classification Guide", "Intermediate", 82),
        ("Semantic Search with AI", "Advanced", 73),
        ("Explainable AI Methods", "Intermediate", 80),
        ("AI Ethics and Bias", "Beginner", 88),
        ("Generative Models Overview", "Intermediate", 87),
        ("Diffusion Models Explained", "Advanced", 69),
        ("Large Language Models Guide", "Intermediate", 90),
        ("Fine-tuning LLMs", "Advanced", 76),
        ("Clustering Algorithms", "Intermediate", 78),
        ("Anomaly Detection", "Intermediate", 77),
        ("GAN Fundamentals", "Advanced", 71),
        ("Model Deployment with AI", "Intermediate", 81),
        ("AlexNet to ResNet Evolution", "Intermediate", 76),
        ("Convolution Operations Deep Dive", "Advanced", 72),
        ("Batch Normalization Explained", "Intermediate", 79),
        ("Dropout and Regularization", "Intermediate", 80),
    ],
    "Web Dev": [
        ("HTML and CSS Foundations", "Beginner", 92),
        ("JavaScript Core Concepts", "Beginner", 91),
        ("React for Beginners", "Intermediate", 89),
        ("Backend APIs with FastAPI", "Intermediate", 85),
        ("Auth and Security Basics", "Intermediate", 83),
        ("Fullstack Project Architecture", "Advanced", 78),
        ("Frontend Performance Tuning", "Advanced", 77),
        ("TypeScript Essentials", "Intermediate", 86),
        ("Vue.js Fundamentals", "Intermediate", 84),
        ("Angular Advanced Patterns", "Advanced", 75),
        ("REST API Design", "Intermediate", 82),
        ("GraphQL Essentials", "Intermediate", 80),
        ("WebSocket Real-time Apps", "Advanced", 74),
        ("CSS Grid and Flexbox", "Beginner", 89),
        ("Responsive Web Design", "Beginner", 87),
        ("SEO for Web Developers", "Intermediate", 81),
        ("Web Accessibility Standards", "Intermediate", 78),
        ("JavaScript Testing Guide", "Intermediate", 83),
        ("React Hooks Deep Dive", "Advanced", 81),
        ("State Management Redux", "Intermediate", 79),
        ("Next.js Full Stack", "Advanced", 77),
        ("Docker for Developers", "Intermediate", 82),
        ("Kubernetes Basics", "Advanced", 76),
        ("CI/CD Pipelines", "Intermediate", 80),
        ("Microservices Architecture", "Advanced", 75),
        ("Database Design Principles", "Intermediate", 79),
        ("MongoDB NoSQL Guide", "Intermediate", 77),
        ("PostgreSQL Advanced Queries", "Intermediate", 81),
        ("Caching Strategies", "Intermediate", 78),
        ("Load Balancing Techniques", "Advanced", 74),
        ("Web Security Best Practices", "Intermediate", 85),
        ("OAuth2 Implementation", "Advanced", 73),
        ("API Rate Limiting", "Intermediate", 76),
        ("Error Handling in APIs", "Intermediate", 80),
    ],
    "Data Science": [
        ("Statistics for Data Analysis", "Beginner", 89),
        ("Pandas Workflow Mastery", "Intermediate", 86),
        ("Data Visualization with Matplotlib", "Beginner", 85),
        ("Feature Engineering Techniques", "Advanced", 79),
        ("Recommendation Systems Design", "Advanced", 82),
        ("A/B Testing for Product Teams", "Intermediate", 81),
        ("SQL for Analysts", "Beginner", 90),
        ("Advanced SQL Queries", "Intermediate", 84),
        ("Exploratory Data Analysis", "Beginner", 87),
        ("Data Cleaning Best Practices", "Beginner", 88),
        ("Statistical Hypothesis Testing", "Intermediate", 80),
        ("Regression Analysis Guide", "Intermediate", 82),
        ("Classification Models", "Intermediate", 81),
        ("Dimensionality Reduction", "Advanced", 76),
        ("Ensemble Methods", "Advanced", 77),
        ("Time Series Forecasting", "Advanced", 75),
        ("Anomaly Detection Techniques", "Intermediate", 78),
        ("Causal Inference Basics", "Advanced", 72),
        ("Bayesian Statistics", "Advanced", 71),
        ("Probability Distributions", "Intermediate", 79),
        ("Data Pipeline Design", "Intermediate", 80),
        ("ETL Processes", "Intermediate", 79),
        ("Data Warehousing", "Advanced", 74),
        ("Big Data with Spark", "Advanced", 73),
        ("Apache Kafka Streaming", "Advanced", 72),
        ("Machine Learning Ops", "Intermediate", 81),
        ("Model Monitoring", "Intermediate", 80),
        ("Feature Stores", "Advanced", 75),
        ("Experiment Tracking", "Intermediate", 82),
        ("Visualization with Tableau", "Intermediate", 83),
        ("Seaborn Advanced Plots", "Intermediate", 81),
        ("Plotly Interactive Dashboards", "Intermediate", 80),
        ("Power BI Essentials", "Beginner", 84),
        ("Data Storytelling", "Beginner", 86),
    ],
}

INTERACTION_TYPES = ["view", "click", "like", "complete"]


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def reset_existing_data(db) -> None:
    """Clear existing rows so each run recreates the intended dataset shape."""
    db.query(Interaction).delete(synchronize_session=False)
    db.query(UserSkill).delete(synchronize_session=False)
    db.query(ContentSkill).delete(synchronize_session=False)
    db.query(User).delete(synchronize_session=False)
    db.query(Content).delete(synchronize_session=False)
    db.query(Skill).delete(synchronize_session=False)
    db.commit()


def seed_users(db) -> tuple[list[User], int]:
    """Create 40 users with category preferences and skills."""
    existing_users = {u.name: u for u in db.query(User).all()}
    inserted = 0
    updated = 0

    for category, names in USER_PROFILES.items():
        for name in names:
            # Build interests string: category + top 3 skills
            top_skills = CATEGORY_SKILLS[category][:3]
            expected_interests = f"{category}, {', '.join(top_skills)}"

            if name in existing_users:
                existing_user = existing_users[name]
                if existing_user.interests != expected_interests:
                    existing_user.interests = expected_interests
                    updated += 1
                continue

            user = User(name=name, interests=expected_interests)
            db.add(user)
            inserted += 1

    db.commit()
    if updated:
        print(f"Updated existing user profiles: {updated}")
    users = db.query(User).order_by(User.id.asc()).all()
    return users, inserted


def seed_content(db) -> tuple[list[Content], int]:
    """Create moderate content catalog (72 items) with category-balanced coverage."""
    existing_content = {c.title: c for c in db.query(Content).all()}
    inserted = 0

    for category, items in CONTENT_TEMPLATES.items():
        for title, difficulty, base_popularity in items[:CONTENT_ITEMS_PER_CATEGORY]:
            if title in existing_content:
                continue

            # Add some noise to popularity to create variety
            popularity = base_popularity + RNG.randint(-5, 5)
            popularity = max(50, min(100, popularity))  # Keep in [50, 100]

            content = Content(
                title=title,
                category=category,
                difficulty=difficulty,
                popularity=popularity,
            )
            db.add(content)
            inserted += 1

    db.commit()
    content_items = db.query(Content).order_by(Content.id.asc()).all()
    return content_items, inserted


def seed_skills(db) -> tuple[list[Skill], int]:
    existing_skills = {s.name: s for s in db.query(Skill).all()}
    inserted = 0

    for name in SKILL_NAMES:
        if name in existing_skills:
            continue

        db.add(Skill(name=name))
        inserted += 1

    db.commit()
    skills = db.query(Skill).order_by(Skill.id.asc()).all()
    return skills, inserted


def seed_user_skills(db, users: list[User], skills: list[Skill]) -> int:
    """Assign skills to users based on their category preferences."""
    existing_pairs = {(us.user_id, us.skill_id) for us in db.query(UserSkill).all()}
    inserted = 0
    skills_by_name = {skill.name: skill for skill in skills}

    for user in users:
        # Find user's preferred category
        category = USER_CATEGORY_MAP.get(user.name)
        if not category:
            continue

        # Assign primary skills from user's category
        primary_skill_names = CATEGORY_SKILLS[category][:5]
        secondary_skill_names = [
            skill for cat in CATEGORIES if cat != category
            for skill in CATEGORY_SKILLS[cat][:1]
        ]

        selected_skill_names = primary_skill_names + secondary_skill_names

        for skill_name in selected_skill_names:
            if skill_name not in skills_by_name:
                continue

            skill = skills_by_name[skill_name]
            key = (user.id, skill.id)
            if key in existing_pairs:
                continue

            # Proficiency: higher for primary skills
            is_primary = skill_name in primary_skill_names
            proficiency = RNG.randint(4, 5) if is_primary else RNG.randint(2, 3)

            db.add(UserSkill(user_id=user.id, skill_id=skill.id, proficiency=proficiency))
            existing_pairs.add(key)
            inserted += 1

    db.commit()
    return inserted


def seed_content_skills(db, content_items: list[Content], skills: list[Skill]) -> int:
    existing_pairs = {(cs.content_id, cs.skill_id) for cs in db.query(ContentSkill).all()}
    inserted = 0
    skills_by_name = {skill.name: skill for skill in skills}

    for item in content_items:
        category_skills = CATEGORY_SKILLS.get(item.category, [])
        selected_names = category_skills[:3]
        selected_skills = [skills_by_name[name] for name in selected_names if name in skills_by_name]

        for skill in selected_skills:
            key = (item.id, skill.id)
            if key in existing_pairs:
                continue

            db.add(ContentSkill(content_id=item.id, skill_id=skill.id))
            existing_pairs.add(key)
            inserted += 1

    db.commit()
    return inserted


def seed_interactions(db, users: list[User], content_items: list[Content]) -> int:
    """Create strong collaborative overlap with shared category core pools."""
    existing_interactions = {
        (i.user_id, i.content_id)
        for i in db.query(Interaction.user_id, Interaction.content_id).all()
    }
    inserted = 0

    # Group content by category.
    content_by_category: dict[str, list[Content]] = {category: [] for category in CATEGORIES}
    for item in content_items:
        if item.category in content_by_category:
            content_by_category[item.category].append(item)

    # Sort by popularity (descending) and id for deterministic selection.
    for category in CATEGORIES:
        content_by_category[category].sort(key=lambda c: (-int(c.popularity), c.id))

    core_pool_by_category: dict[str, list[Content]] = {}
    non_core_pool_by_category: dict[str, list[Content]] = {}
    mandatory_overlap_by_category: dict[str, list[Content]] = {}
    for category in CATEGORIES:
        items = content_by_category.get(category, [])
        core_pool = items[: min(CORE_POOL_SIZE, len(items))]
        non_core_pool = items[len(core_pool) :]
        mandatory_overlap = core_pool[: min(MANDATORY_SHARED_COUNT, len(core_pool))]

        core_pool_by_category[category] = core_pool
        non_core_pool_by_category[category] = non_core_pool
        mandatory_overlap_by_category[category] = mandatory_overlap

    # Create interactions per user with strong in-category overlap.
    for user in users:
        preferred_category = USER_CATEGORY_MAP.get(user.name)
        if not preferred_category:
            continue

        preferred_items = content_by_category.get(preferred_category, [])
        preferred_core_pool = core_pool_by_category.get(preferred_category, [])
        preferred_non_core_pool = non_core_pool_by_category.get(preferred_category, [])
        mandatory_overlap_items = mandatory_overlap_by_category.get(preferred_category, [])

        other_items = [
            item for category, items in content_by_category.items()
            if category != preferred_category
            for item in items
        ]

        if not preferred_items or not preferred_core_pool or not other_items:
            continue

        # User-specific RNG for reproducibility.
        user_rng = random.Random(1000 + user.id)

        # Target 10-15 interactions per user, using feasible totals for 10-15% integer noise.
        num_interactions = user_rng.choice([10, 14, 15])

        # Keep cross-category noise low: 10-15%.
        if num_interactions == 10:
            num_other = 1
        else:
            num_other = 2
        num_preferred = num_interactions - num_other

        # Within preferred category: 70% core shared items, 30% random category items.
        num_core_target = max(1, (7 * num_preferred + 9) // 10)  # ceil(70%)
        num_random_target = max(0, num_preferred - num_core_target)

        # Mandatory overlap guarantees at least 5-8 shared items among same-category users.
        mandatory_take = min(len(mandatory_overlap_items), num_core_target)
        selected_core = list(mandatory_overlap_items[:mandatory_take])

        # Expand core picks from remaining core pool to reach core target.
        if len(selected_core) < num_core_target:
            selected_ids = {item.id for item in selected_core}
            core_remainder = [item for item in preferred_core_pool if item.id not in selected_ids]
            take = min(num_core_target - len(selected_core), len(core_remainder))
            if take > 0:
                selected_core.extend(user_rng.sample(core_remainder, k=take))

        # Pick non-core preferred items for diversity.
        selected_preferred_random: list[Content] = []
        if num_random_target > 0:
            if len(preferred_non_core_pool) >= num_random_target:
                selected_preferred_random = user_rng.sample(preferred_non_core_pool, k=num_random_target)
            else:
                selected_preferred_random = preferred_non_core_pool.copy()

        # Top up preferred picks if needed.
        selected_preferred = selected_core + selected_preferred_random
        if len(selected_preferred) < num_preferred:
            selected_ids = {item.id for item in selected_preferred}
            fallback_pool = [item for item in preferred_items if item.id not in selected_ids]
            extra_take = min(num_preferred - len(selected_preferred), len(fallback_pool))
            if extra_take > 0:
                selected_preferred.extend(user_rng.sample(fallback_pool, k=extra_take))

        selected_other = user_rng.sample(other_items, k=min(num_other, len(other_items)))

        # Final safety: enforce exactly num_interactions unique items.
        combined_unique: dict[int, tuple[Content, str]] = {}
        selected_core_ids = {x.id for x in selected_core}
        for item in selected_preferred:
            label = "core" if item.id in selected_core_ids else "preferred_random"
            combined_unique[item.id] = (item, label)
        for item in selected_other:
            combined_unique[item.id] = (item, "cross_category")

        if len(combined_unique) < num_interactions:
            selected_ids = set(combined_unique.keys())
            all_pool = [item for item in preferred_items + other_items if item.id not in selected_ids]
            top_up = min(num_interactions - len(combined_unique), len(all_pool))
            for item in user_rng.sample(all_pool, k=top_up):
                label = "preferred_random" if item.category == preferred_category else "cross_category"
                combined_unique[item.id] = (item, label)

        # Trim deterministically if we overshoot.
        final_items = list(combined_unique.values())[:num_interactions]

        # Keep one deterministic user-item pair free for API feedback integration tests.
        if user.id == 1:
            final_items = [(item, kind) for item, kind in final_items if item.id != 1]
            if len(final_items) < num_interactions:
                selected_ids = {item.id for item, _ in final_items}
                preferred_fallback_pool = [
                    item for item in preferred_items if item.id not in selected_ids and item.id != 1
                ]
                cross_fallback_pool = [
                    item
                    for item in other_items
                    if item.id not in selected_ids and item.id != 1
                ]
                fallback_pool = preferred_fallback_pool + cross_fallback_pool
                extra_take = min(num_interactions - len(final_items), len(fallback_pool))
                if extra_take > 0:
                    chosen: list[Content] = []
                    take_pref = min(extra_take, len(preferred_fallback_pool))
                    if take_pref > 0:
                        chosen.extend(user_rng.sample(preferred_fallback_pool, k=take_pref))

                    remaining = extra_take - len(chosen)
                    if remaining > 0:
                        chosen.extend(user_rng.sample(cross_fallback_pool, k=remaining))

                    for item in chosen:
                        label = "preferred_random" if item.category == preferred_category else "cross_category"
                        final_items.append((item, label))

        # Ensure at least 70% of preferred-category interactions come from core shared items.
        preferred_positions = [
            idx for idx, (item, _kind) in enumerate(final_items) if item.category == preferred_category
        ]
        preferred_count = len(preferred_positions)
        if preferred_count > 0:
            required_core_count = (7 * preferred_count + 9) // 10  # ceil(70%)
            current_core_count = sum(
                1
                for idx in preferred_positions
                if final_items[idx][1] == "core"
            )

            if current_core_count < required_core_count:
                selected_ids = {item.id for item, _ in final_items}
                available_core = [
                    item
                    for item in preferred_core_pool
                    if item.id not in selected_ids and not (user.id == 1 and item.id == 1)
                ]
                replace_positions = [
                    idx for idx in preferred_positions if final_items[idx][1] != "core"
                ]
                replace_take = min(
                    required_core_count - current_core_count,
                    len(available_core),
                    len(replace_positions),
                )

                for i in range(replace_take):
                    final_items[replace_positions[i]] = (available_core[i], "core")

        # Create interactions. High ratings are concentrated on core shared items.
        for item, item_type in final_items:
            key = (user.id, item.id)
            if key in existing_interactions:
                continue

            if item_type == "core":
                rating = round(user_rng.uniform(4.2, 5.0), 1)
                interaction_type = "complete" if rating >= 4.6 else "like"
            elif item_type == "preferred_random":
                rating = round(user_rng.uniform(3.0, 4.1), 1)
                interaction_type = "like" if rating >= 3.8 else user_rng.choice(["click", "view"])
            else:
                rating = round(user_rng.uniform(1.0, 3.0), 1)
                interaction_type = user_rng.choice(["view", "click"])

            db.add(
                Interaction(
                    user_id=user.id,
                    content_id=item.id,
                    type=interaction_type,
                    rating=rating,
                )
            )
            existing_interactions.add(key)
            inserted += 1

    db.commit()
    return inserted


def print_summary(db, inserted_counts: dict[str, int]) -> None:
    """Print comprehensive dataset summary."""
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_content = db.query(func.count(Content.id)).scalar() or 0
    total_skills = db.query(func.count(Skill.id)).scalar() or 0
    total_user_skills = db.query(func.count(UserSkill.user_id)).scalar() or 0
    total_content_skills = db.query(func.count(ContentSkill.content_id)).scalar() or 0
    total_interactions = db.query(func.count(Interaction.user_id)).scalar() or 0

    print("\n" + "=" * 70)
    print("SEEDING COMPLETE - DATASET SUMMARY")
    print("=" * 70)

    print("\nRecords Inserted:")
    print(f"  {inserted_counts['users']:3d} users")
    print(f"  {inserted_counts['content']:3d} content items")
    print(f"  {inserted_counts['skills']:3d} skills")
    print(f"  {inserted_counts['user_skills']:3d} user-skill associations")
    print(f"  {inserted_counts['content_skills']:3d} content-skill associations")
    print(f"  {inserted_counts['interactions']:3d} interactions")

    print("\nTotal Records in Database:")
    print(f"  {total_users:3d} users")
    print(f"  {total_content:3d} content items")
    print(f"  {total_skills:3d} skills")
    print(f"  {total_user_skills:3d} user-skill associations")
    print(f"  {total_content_skills:3d} content-skill associations")
    print(f"  {total_interactions:3d} total interactions")

    # Category distribution
    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION")
    print("-" * 70)

    users_per_category = Counter()
    content_per_category = Counter()
    interactions_per_category = Counter()

    # Users by category
    users = db.query(User).all()
    for user in users:
        category = USER_CATEGORY_MAP.get(user.name)
        if category:
            users_per_category[category] += 1

    # Content by category
    for item in db.query(Content).all():
        if item.category in CATEGORIES:
            content_per_category[item.category] += 1

    # Interactions by content category
    interactions_by_cat = (
        db.query(Content.category, func.count(Interaction.user_id))
        .join(Interaction, Interaction.content_id == Content.id)
        .group_by(Content.category)
        .all()
    )
    for category, count in interactions_by_cat:
        interactions_per_category[category] = count

    print("Users per category:")
    for category in sorted(CATEGORIES):
        count = users_per_category.get(category, 0)
        pct = (count / total_users * 100) if total_users > 0 else 0
        print(f"  {category:15s}: {count:3d} users ({pct:5.1f}%)")

    print("\nContent per category:")
    for category in sorted(CATEGORIES):
        count = content_per_category.get(category, 0)
        pct = (count / total_content * 100) if total_content > 0 else 0
        print(f"  {category:15s}: {count:3d} items ({pct:5.1f}%)")

    print("\nInteractions by content category:")
    for category in sorted(CATEGORIES):
        count = interactions_per_category.get(category, 0)
        pct = (count / total_interactions * 100) if total_interactions > 0 else 0
        print(f"  {category:15s}: {count:3d} interactions ({pct:5.1f}%)")

    # Per-user interaction statistics
    print("\n" + "-" * 70)
    print("INTERACTION STATISTICS")
    print("-" * 70)

    per_user_interactions = (
        db.query(Interaction.user_id, func.count(Interaction.user_id))
        .group_by(Interaction.user_id)
        .all()
    )
    if per_user_interactions:
        interactions_list = [count for _, count in per_user_interactions]
        avg_interactions = sum(interactions_list) / len(interactions_list)
        min_interactions = min(interactions_list)
        max_interactions = max(interactions_list)
        print(f"Per-user interactions:")
        print(f"  Min:  {min_interactions} interactions")
        print(f"  Avg:  {avg_interactions:.1f} interactions")
        print(f"  Max:  {max_interactions} interactions")

    # Rating statistics
    print("\nRating distribution:")
    avg_rating = db.query(func.avg(Interaction.rating)).scalar() or 0
    min_rating = db.query(func.min(Interaction.rating)).scalar() or 0
    max_rating = db.query(func.max(Interaction.rating)).scalar() or 0
    print(f"  Min rating: {min_rating}")
    print(f"  Avg rating: {avg_rating:.2f}")
    print(f"  Max rating: {max_rating}")

    # Category preference adherence: check low cross-category noise (10-15%).
    print("\nCategory preference adherence (target 85-90% from preferred):")
    for user in users[:5]:  # Sample 5 users
        preferred_cat = USER_CATEGORY_MAP.get(user.name)
        if not preferred_cat:
            continue

        user_interactions = (
            db.query(Interaction)
            .filter(Interaction.user_id == user.id)
            .all()
        )
        if not user_interactions:
            continue

        preferred_count = 0
        for interaction in user_interactions:
            content = db.query(Content).filter(Content.id == interaction.content_id).first()
            if content and content.category == preferred_cat:
                preferred_count += 1

        pct = (preferred_count / len(user_interactions) * 100) if user_interactions else 0
        print(f"  {user.name:15s}: {preferred_count}/{len(user_interactions)} from {preferred_cat:12s} ({pct:5.1f}%)")

    # Sample interactions
    print("\n" + "-" * 70)
    print("SAMPLE INTERACTIONS (First 15)")
    print("-" * 70)

    sample_rows = (
        db.query(User.name, Content.title, Content.category, Interaction.rating)
        .join(Interaction, Interaction.user_id == User.id)
        .join(Content, Content.id == Interaction.content_id)
        .order_by(User.id.asc(), Interaction.created_at.asc())
        .limit(15)
        .all()
    )

    for user_name, title, category, rating in sample_rows:
        user_pref_cat = USER_CATEGORY_MAP.get(user_name)
        rating_type = "preferred" if category == user_pref_cat else "other"
        print(f"  {user_name:15s} → {title:40s} ({category:12s}) rating={rating:4.1f} [{rating_type}]")

    print("\n" + "=" * 70 + "\n")


def main() -> None:
    create_tables()

    with SessionLocal() as db:
        reset_existing_data(db)

        users, users_inserted = seed_users(db)
        content_items, content_inserted = seed_content(db)
        skills, skills_inserted = seed_skills(db)

        user_skills_inserted = seed_user_skills(db, users, skills)
        content_skills_inserted = seed_content_skills(db, content_items, skills)
        interactions_inserted = seed_interactions(db, users, content_items)

        print_summary(
            db,
            {
                "users": users_inserted,
                "content": content_inserted,
                "skills": skills_inserted,
                "user_skills": user_skills_inserted,
                "content_skills": content_skills_inserted,
                "interactions": interactions_inserted,
            },
        )


if __name__ == "__main__":
    main()
