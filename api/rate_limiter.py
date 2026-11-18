import os

import redis
from dotenv import load_dotenv
from fastapi import HTTPException, Request

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT_MAX_REQUESTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60

# Persistent Redis client
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


def check_rate_limit(request: Request) -> None:
    """Enforces a fixed-window rate limit per client IP using Redis."""
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_ip}"

    try:
        current_count = redis_client.incr(key)
        if current_count == 1:
            redis_client.expire(key, RATE_LIMIT_WINDOW_SECONDS)

        if current_count > RATE_LIMIT_MAX_REQUESTS:
            ttl = redis_client.ttl(key)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded ({RATE_LIMIT_MAX_REQUESTS} requests per "
                    f"{RATE_LIMIT_WINDOW_SECONDS}s). Try again in {max(ttl, 1)}s."
                ),
            )
    except redis.RedisError:
        # Fail-open if Redis is temporarily unreachable
        pass


def get_cached_value(key: str) -> str | None:
    """Fetches a cached string value from Redis, returning None on miss/error."""
    try:
        return redis_client.get(f"cache:{key}")
    except redis.RedisError:
        return None


def set_cached_value(key: str, value: str, ttl_seconds: int = 300) -> None:
    """Stores a value in Redis with a Time-To-Live (default 5 minutes)."""
    try:
        redis_client.setex(f"cache:{key}", ttl_seconds, value)
    except redis.RedisError:
        pass
