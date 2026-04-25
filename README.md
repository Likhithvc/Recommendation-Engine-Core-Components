# AI-Powered Recommendation System with API

## Project Overview

This project is an end-to-end recommendation platform that generates personalized content suggestions through a modular engine and serves them via REST APIs.

The system is inspired by real-world recommendation ecosystems used by platforms like Netflix and Amazon, where user behavior, item similarity, and popularity signals are combined to deliver relevant recommendations.

---

## Architecture

### Recommendation Pipeline

```text
User -> API -> Orchestrator -> Engine -> Database -> Response
```

### Component Breakdown

- User: Sends recommendation and feedback requests.
- API (FastAPI): Exposes endpoints, validates input, applies authentication, and returns structured JSON responses.
- Orchestrator: Coordinates data loading, candidate generation, scoring, caching, and explanation generation.
- Engine: Core recommendation logic (similarity, candidate generation, scoring, ranking, evaluation).
- Database (SQLite + SQLAlchemy): Stores users, content, skills, and interactions.
- Response: Returns ranked recommendations with scores and explanation reasons.

---

## Features

- Personalized recommendations based on user interaction history.
- Hybrid recommendation strategy:
  - Collaborative filtering
  - Content-based filtering
  - Popularity-based ranking
- Cold start handling for new users.
- In-memory caching with TTL for faster repeated recommendation requests.
- REST API for recommendations, feedback, health, and metrics.
- Offline evaluation pipeline with Precision@5, Recall@5, and NDCG@5.

---

## Tech Stack

- Python
- FastAPI
- SQLite
- SQLAlchemy
- NumPy / scikit-learn

---

## Project Structure

```text
.
|-- api/
|   |-- app.py                 # FastAPI application and endpoints
|-- data/
|   |-- database.py            # SQLAlchemy engine, session, Base
|   |-- models.py              # ORM models
|   |-- repositories.py        # Data access layer
|-- engine/
|   |-- similarity.py          # Similarity calculations
|   |-- candidate_gen.py       # Candidate generation logic
|   |-- scorer.py              # Weighted scoring and ranking
|   |-- orchestrator.py        # End-to-end recommendation orchestration
|   |-- evaluator.py           # Evaluation metrics
|-- scripts/
|   |-- seed_data.py           # Deterministic dataset seeding
|   |-- evaluate.py            # Offline evaluation script
|-- tests/
|   |-- test_api.py            # API tests
|   |-- test_engine.py         # Orchestrator and caching tests
|   |-- test_data.py           # Data/seed integrity tests
|-- requirements.txt
|-- pytest.ini
|-- evaluation_report.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "AI-Powered Recommendation System with API"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Seed the Database

```bash
python scripts/seed_data.py
```

### 4. Start the API Server

```bash
uvicorn api.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

## API Endpoints

> Note: All endpoints except `/health` require the `x-api-key` header.

### `GET /recommend/{user_id}`
Returns personalized recommendations with score and reason.

### `POST /feedback`
Stores user feedback interaction (`user_id`, `content_id`, optional `rating`).

### `GET /health`
Returns service health status.

### `GET /metrics`
Returns API metrics such as total requests, average response time, and cache hit rate.

---

## Evaluation Results

Offline evaluation results (Top-K, K=5):

- Precision@5: 0.36
- Recall@5: 0.70
- NDCG@5: 0.58

### Interpretation

- Precision@5 (0.36): A meaningful proportion of the top recommendations are relevant.
- Recall@5 (0.70): The system retrieves most relevant items well.
- NDCG@5 (0.58): Ranking quality is good but can still be improved.

Overall, the system demonstrates strong retrieval ability with room for better ranking calibration.

---

## Future Improvements

- Introduce advanced ML recommenders (e.g., matrix factorization, neural ranking).
- Add embedding-based user/item representations.
- Scale for large datasets and high-throughput deployment.
- Enhance feature engineering with temporal/contextual behavior signals.
- Add online A/B testing and model monitoring.

---

## Conclusion

This AI-powered recommendation system provides a practical and modular foundation for personalized content delivery. It integrates data modeling, hybrid recommendation logic, caching, explainable outputs, API serving, and evaluation in a cohesive architecture suitable for learning, prototyping, and further research.
