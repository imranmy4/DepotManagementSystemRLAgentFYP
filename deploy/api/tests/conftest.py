"""Fixtures for database tests: a fresh `depot_test` database built by the Alembic migrations,
and a per-test session that is rolled back afterwards.

Needs the Postgres container running (cd deploy && docker compose up -d); skipped otherwise.
"""
import os

# Must be set before app.settings is imported, so nothing can touch the real database.
os.environ["POSTGRES_DB"] = os.environ.get("DEPOT_TEST_DB", "depot_test")

from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.settings import settings
from depot.env import DepotConfig

assert settings.postgres_db.endswith("_test"), "tests must run against a *_test database"

API_DIR = Path(__file__).resolve().parents[1]

# Small yard: 2 blocks side by side, 2 bays x 2 rows x 3 tiers each.
TINY = DepotConfig(n_blocks=2, block_rows=1, n_bays=2, n_rows=2, n_tiers=3)


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
