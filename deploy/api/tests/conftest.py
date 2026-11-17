"""Fixtures: a fresh `depot_test` database built by the Alembic migrations, a per-test
session that is rolled back afterwards, a committed database for worker tests, and Redis db 15.

Needs the Postgres and Redis containers running (cd deploy && docker compose up -d);
dependent tests are skipped otherwise.
"""
import os

# Must be set before app.settings is imported, so nothing can touch the real database / queue.
os.environ["POSTGRES_DB"] = os.environ.get("DEPOT_TEST_DB", "depot_test")
os.environ["REDIS_URL"] = os.environ.get("DEPOT_TEST_REDIS_URL", "redis://localhost:6379/15")

from pathlib import Path

import psycopg
import pytest
import redis
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.settings import settings
from helpers import TINY

assert settings.postgres_db.endswith("_test"), "tests must run against a *_test database"
assert settings.redis_url.endswith("/15"), "tests must run against Redis db 15"

API_DIR = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def engine():
    admin = settings.database_url.set(drivername="postgresql", database="postgres")
    try:
        conn = psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True,
                               connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable: {e}")
    with conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{settings.postgres_db}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{settings.postgres_db}"')

    command.upgrade(Config(str(API_DIR / "alembic.ini")), "head")
    eng = create_engine(settings.database_url)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    """Session whose commits stay inside one outer transaction that is rolled back."""
    with engine.connect() as conn:
        outer = conn.begin()
        with Session(bind=conn, join_transaction_mode="create_savepoint") as s:
            yield s
        outer.rollback()


@pytest.fixture
def yard(session):
    """Session with the TINY layout seeded."""
    from app.db.seed import seed_layout
    seed_layout(session, TINY)
    return session


@pytest.fixture
def committed(engine):
    """Database with the TINY layout where code commits for real (worker tests); emptied after."""
    from app.db.seed import seed_layout
    from app.db.session import SessionLocal
    with SessionLocal.begin() as s:
        seed_layout(s, TINY)
    yield SessionLocal
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE move_log, orders, container_location, container, yard_state, "
                          "slot, yard_block RESTART IDENTITY CASCADE"))


@pytest.fixture
def r():
    """Empty Redis db 15."""
    from app import order_queue
    client = order_queue.connect()
    try:
        client.ping()
    except redis.ConnectionError as e:
        pytest.skip(f"Redis not reachable: {e}")
    client.flushdb()
    yield client
    client.flushdb()
