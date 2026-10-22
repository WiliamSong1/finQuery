
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    finquery_api_key: str = ""
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "FinQuery API"
    environment: str = "development"

    qdrant_path: Path = (
        PROJECT_ROOT / "data" / "qdrant"
    )

    max_upload_mb: int = Field(default=25, ge=1, le=100)

    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-20b"

    sec_user_agent: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
