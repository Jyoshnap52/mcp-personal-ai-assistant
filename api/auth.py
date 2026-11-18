import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid

import psycopg
from dotenv import load_dotenv
from fastapi import Cookie, HTTPException, Response

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://assistant:assistant@localhost:5433/assistant",
)
# Secret for signing cookies (uses env var or generates an ephemeral key per run)
SESSION_SECRET = os.getenv("SESSION_SECRET") or os.getenv("GROQ_API_KEY") or secrets.token_hex(32)
COOKIE_NAME = "mcp_session"
SESSION_TTL_SECONDS = 86400 * 7  # 7 days


def _get_conn():
    return psycopg.connect(DATABASE_URL, autocommit=True)


def init_users_table() -> None:
    """Creates the users table in PostgreSQL if it does not exist."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)


init_users_table()


def hash_password(password: str) -> str:
    """Hashes a password with a random 16-byte salt using scrypt."""
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return f"{base64.b64encode(salt).decode()}${base64.b64encode(derived).decode()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a password in constant time against a stored scrypt hash."""
    try:
        salt_b64, hash_b64 = stored_hash.split("$", 1)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def _sign_session(user_id: str, email: str) -> str:
    payload = json.dumps(
        {
            "uid": user_id,
            "email": email,
            "exp": int(time.time()) + SESSION_TTL_SECONDS,
        }
    )
    b64 = base64.urlsafe_b64encode(payload.encode("utf-8")).decode()
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _verify_session(token: str) -> dict | None:
    try:
        b64, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(
            SESSION_SECRET.encode("utf-8"), b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        data = json.loads(base64.urlsafe_b64decode(b64.encode("utf-8")).decode())
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def set_session_cookie(response: Response, user_id: str, email: str) -> None:
    token = _sign_session(user_id, email)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )


def create_user(email: str, password: str) -> dict[str, str]:
    clean_email = email.strip().lower()
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (id, email, password_hash) VALUES (%s, %s, %s);",
                    (user_id, clean_email, pw_hash),
                )
    except psycopg.errors.UniqueViolation:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        ) from None
    return {"user_id": user_id, "email": clean_email}


def authenticate_user(email: str, password: str) -> dict[str, str]:
    clean_email = email.strip().lower()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, password_hash FROM users WHERE email = %s;",
                (clean_email,),
            )
            row = cur.fetchone()
    if not row or not verify_password(password, row[2]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"user_id": str(row[0]), "email": row[1]}


def get_current_user(mcp_session: str | None = Cookie(default=None)) -> dict[str, str]:
    """Dependency that validates the HttpOnly session cookie on protected routes."""
    if not mcp_session:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    data = _verify_session(mcp_session)
    if not data:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please sign in.")
    return {"user_id": data["uid"], "email": data["email"]}
