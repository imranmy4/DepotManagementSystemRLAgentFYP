"""Deployment settings read from deploy/.env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

DEPLOY_DIR = Path(__file__).resolve().parents[2]          # .../deploy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=DEPLOY_DIR / ".env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    redis_url: str = "redis://localhost:6379/0"

    # Real seconds per model tick, the unit of dwell time the model was trained on (1 day).
    tick_seconds: int = 86400

    @property
    def database_url(self) -> URL:
        # URL.create escapes special characters in the password (e.g. @ or %).
        return URL.create("postgresql+psycopg", username=self.postgres_user,
                          password=self.postgres_password, host=self.postgres_host,
                          port=self.postgres_port, database=self.postgres_db)


settings = Settings()
