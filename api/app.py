"""FastAPI application for recommendation and feedback APIs."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import suppress

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from data.database import SessionLocal
from data.models import Content
from data.repositories import InteractionRepository, UserRepository
from engine.orchestrator import RecommendationOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | request_id=%(request_id)s | %(message)s",
)
LOGGER = logging.getLogger("recommendation_api")
API_KEY = "dev-api-key-123"


class RequestIDFilter(logging.Filter):
    """Ensures log records always include request_id for formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


LOGGER.addFilter(RequestIDFilter())


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    content_id: int = Field(..., gt=0)
    rating: float | None = Field(default=None, ge=0.0, le=5.0)


class FeedbackResponse(BaseModel):
    request_id: str
    message: str


class RecommendationItem(BaseModel):
    content_id: int
    score: float
    reason: str
    title: str | None = None
    explanation: str | None = None


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: int
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    service: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_response_time: float
    cache_hit_rate: float


class _InMemoryMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_response_time_ms = 0.0
        self.cache_checks = 0
        self.cache_hits = 0

    def record(self, duration_ms: float, cache_hit: bool | None = None) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_response_time_ms += duration_ms

            if cache_hit is not None:
                self.cache_checks += 1
                if cache_hit:
                    self.cache_hits += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            avg_ms = (
                self.total_response_time_ms / self.total_requests
                if self.total_requests > 0
                else 0.0
            )
            cache_hit_rate = (
                self.cache_hits / self.cache_checks if self.cache_checks > 0 else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "avg_response_time": round(avg_ms, 2),
                "cache_hit_rate": round(cache_hit_rate, 4),
            }


class _RecommendationCache:
    """Simple in-memory TTL cache for recommendations."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._lock = threading.Lock()
        self._store: dict[tuple[int, int], dict[str, Any]] = {}
        self.ttl_seconds = max(1, int(ttl_seconds))

    def get(self, user_id: int, limit: int) -> list[dict[str, Any]] | None:
        key = (user_id, limit)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            if entry["expires_at"] <= now:
                del self._store[key]
                return None
            return [dict(item) for item in entry["recommendations"]]

    def set(self, user_id: int, limit: int, recommendations: list[dict[str, Any]]) -> None:
        key = (user_id, limit)
        with self._lock:
            self._store[key] = {
                "recommendations": [dict(item) for item in recommendations],
                "expires_at": time.time() + self.ttl_seconds,
            }


app = FastAPI(title="Recommendation API", version="1.0.0")
metrics = _InMemoryMetrics()
recommendation_cache = _RecommendationCache(ttl_seconds=60)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()

    LOGGER.info(
        "Incoming request %s %s",
        request.method,
        request.url.path,
        extra={"request_id": request_id},
    )

    if request.url.path != "/health":
        provided_key = request.headers.get("x-api-key")
        if provided_key != API_KEY:
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record(duration_ms, cache_hit=None)
            LOGGER.warning(
                "Unauthorized request: invalid or missing API key",
                extra={"request_id": request_id},
            )
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "request_id": request_id,
                    "detail": "Invalid or missing API key. Provide a valid x-api-key header.",
                },
            )
            response.headers["X-Request-ID"] = request_id
            return response

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record(duration_ms, cache_hit=None)
        LOGGER.exception("Unhandled request error", extra={"request_id": request_id})
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    cache_hit = getattr(request.state, "cache_hit", None)
    metrics.record(duration_ms, cache_hit=cache_hit)
    response.headers["X-Request-ID"] = request_id

    LOGGER.info(
        "Completed request %s %s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={"request_id": request_id},
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=exc.status_code,
        content={"request_id": request_id, "detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"request_id": request_id, "detail": exc.errors()},
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(request: Request, user_id: int, limit: int = 5):
    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id must be a positive integer",
        )
    if limit <= 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 100",
        )

    db = SessionLocal()
    orchestrator = RecommendationOrchestrator(db=db)
    try:
        cached_recommendations = recommendation_cache.get(user_id=user_id, limit=limit)
        if cached_recommendations is not None:
            request.state.cache_hit = True
            request_id = getattr(request.state, "request_id", "-")
            return RecommendationResponse(
                request_id=request_id,
                user_id=user_id,
                recommendations=[RecommendationItem(**item) for item in cached_recommendations],
            )

        request.state.cache_hit = False
        user = UserRepository(db).get_user(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )

        recommendations = orchestrator.get_recommendations(user_id=user_id, limit=limit)
        recommendation_cache.set(user_id=user_id, limit=limit, recommendations=recommendations)
        request_id = getattr(request.state, "request_id", "-")
        return RecommendationResponse(
            request_id=request_id,
            user_id=user_id,
            recommendations=[RecommendationItem(**item) for item in recommendations],
        )
    finally:
        with suppress(Exception):
            orchestrator.close()


@app.post("/feedback", response_model=FeedbackResponse)
def post_feedback(request: Request, payload: FeedbackRequest):
    db = SessionLocal()
    try:
        user_repo = UserRepository(db)
        interaction_repo = InteractionRepository(db)

        if user_repo.get_user(payload.user_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {payload.user_id} not found",
            )

        content = db.query(Content).filter(Content.id == payload.content_id).first()
        if content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content {payload.content_id} not found",
            )

        interaction_type = "rate" if payload.rating is not None else "view"
        interaction_repo.record_interaction(
            user_id=payload.user_id,
            content_id=payload.content_id,
            type=interaction_type,
            rating=payload.rating,
        )

        request_id = getattr(request.state, "request_id", "-")
        return FeedbackResponse(request_id=request_id, message="Feedback recorded")

    finally:
        with suppress(Exception):
            db.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="recommendation-api")


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    snapshot = metrics.snapshot()
    return MetricsResponse(**snapshot)
