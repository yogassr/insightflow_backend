
# database.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    database_url: str
    frontend_origin: str = "http://localhost:3000"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-lite-latest"
    adaption_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE, ".env"),
        extra="ignore",
    )


settings = Settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass
