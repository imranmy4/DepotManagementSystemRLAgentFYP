"""Per-request dependencies: a database session and the Redis client."""
from collections.abc import Iterator

from sqlalchemy.orm import Session

from app import order_queue
from app.db.session import SessionLocal

_redis = order_queue.connect()  # one client for the whole API; connects lazily

def get_session() -> Iterator[Session]:
    """Database session for one request; uncommited changes are rolled back when it closes."""
    with SessionLocal() as session:
        yield session

def get_redis():
    return _redis