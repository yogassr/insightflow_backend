# database.py

import importlib.util
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_BASE_DIR = Path(__file__).resolve().parent
_ENV_FILE = _BASE_DIR / ".env"
_DEFAULT_SQLITE_URL = f"sqlite:///{(_BASE_DIR / 'insightflow.db').as_posix()}"


def normalize_database_url(raw_url: str | None) -> str:
    """
    Normalize DATABASE_URL for seamless deployment on Render, Vercel Postgres,
    Neon, Supabase, Railway, or local SQLite/PostgreSQL.
    """
    if not raw_url or not raw_url.strip():
        return _DEFAULT_SQLITE_URL

    url = raw_url.strip().strip('"').strip("'")

    # Render / Heroku style postgres:// -> SQLAlchemy 2.0 postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    # Auto-detect installed PostgreSQL driver (psycopg2 vs psycopg v3)
    has_psycopg2 = importlib.util.find_spec("psycopg2") is not None
    has_psycopg3 = importlib.util.find_spec("psycopg") is not None

    if url.startswith("postgresql://"):
        if not has_psycopg2 and has_psycopg3:
            url = "postgresql+psycopg://" + url[len("postgresql://") :]
    elif url.startswith("postgresql+psycopg2://") and not has_psycopg2 and has_psycopg3:
        url = "postgresql+psycopg://" + url[len("postgresql+psycopg2://") :]
    elif url.startswith("postgresql+psycopg://") and not has_psycopg3 and has_psycopg2:
        url = "postgresql+psycopg2://" + url[len("postgresql+psycopg://") :]

    return url


class Settings(BaseSettings):
    database_url: str = _DEFAULT_SQLITE_URL
    frontend_origin: str = "http://localhost:3000"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-lite-latest"
    adaption_api_key: str | None = None
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE, ".env"),
        extra="ignore",
    )

    @property
    def sqlalchemy_database_url(self) -> str:
        return normalize_database_url(self.database_url)

    @property
    def cors_origins(self) -> list[str]:
        """
        Return normalized list of allowed CORS origins (stripping trailing slashes
        so URLs like 'https://app.vercel.app/' match browser Origin headers).
        """
        origins = {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        }
        if self.frontend_origin:
            for item in self.frontend_origin.split(","):
                cleaned = item.strip().rstrip("/")
                if cleaned:
                    origins.add(cleaned)
        return sorted(origins)


settings = Settings()

_db_url = settings.sqlalchemy_database_url
_engine_kwargs: dict = {
    "pool_pre_ping": True,
}

if _db_url.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Recycle connections every 5 minutes to prevent cloud Postgres idle SSL drops
    _engine_kwargs["pool_recycle"] = 300
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10

engine = create_engine(_db_url, **_engine_kwargs)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass
