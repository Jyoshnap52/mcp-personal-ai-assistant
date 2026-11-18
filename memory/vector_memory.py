import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy loads embedding model once in memory."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


class VectorMemoryStore:
    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self._fallback_memories: list[dict[str, Any]] = []
        self._db_available = False
        if self.db_url:
            self._init_db()

    def _get_connection(self):
        conn = psycopg.connect(self.db_url, autocommit=True)
        register_vector(conn)
        return conn

    def _init_db(self):
        """Creates the pgvector memory table and vector index if not exists."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_memories (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(64) NOT NULL,
                            memory_text TEXT NOT NULL,
                            embedding vector(384) NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS memory_embedding_idx 
                        ON user_memories USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
            self._db_available = True
        except Exception as e:
            print(f"Warning: pgvector DB unavailable ({e}). Running with in-memory memory store.")
            self._db_available = False

    def store_memory(self, user_id: str, memory_text: str) -> None:
        """Embeds text and inserts unique memory record into PostgreSQL with memory fallback."""
        clean_text = memory_text.strip()
        if not clean_text:
            return

        if not self._db_available:
            self._fallback_memories.append({"user_id": user_id, "memory": clean_text})
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Skip if this exact preference already exists for the user
                    cur.execute(
                        "SELECT 1 FROM user_memories WHERE user_id = %s AND memory_text = %s;",
                        (user_id, clean_text),
                    )
                    if cur.fetchone():
                        return

                    model = get_embedding_model()
                    embedding = model.encode(clean_text).tolist()
                    cur.execute(
                        """
                        INSERT INTO user_memories (user_id, memory_text, embedding)
                        VALUES (%s, %s, %s);
                        """,
                        (user_id, clean_text, embedding),
                    )
        except Exception:
            self._fallback_memories.append({"user_id": user_id, "memory": clean_text})

    def retrieve_relevant_memories(
        self, user_id: str, query: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Finds top-k most semantically relevant memories using cosine distance with fallback."""
        if not self._db_available:
            return [m for m in self._fallback_memories if m.get("user_id") == user_id][:limit]

        try:
            model = get_embedding_model()
            query_embedding = model.encode(query).tolist()

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT memory_text, embedding <=> %s:: vector AS distance
                        FROM user_memories
                        WHERE user_id = %s
                        ORDER BY distance ASC
                        LIMIT %s;
                        """,
                        (query_embedding, user_id, limit),
                    )
                    rows = cur.fetchall()

            return [{"memory": row[0], "distance": float(row[1])} for row in rows]
        except Exception:
            return [m for m in self._fallback_memories if m.get("user_id") == user_id][:limit]
