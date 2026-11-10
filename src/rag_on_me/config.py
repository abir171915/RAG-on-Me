"""Runtime settings. Each field can be overridden by an environment variable of the
same name in upper case (e.g. TOP_K=8), or by a line in the project's .env file."""
from pathlib import Path
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rag_on_me/config.py -> project root. Only valid for an editable install;
# deployments should set DATA_DIR explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    groq_api_key: Optional[SecretStr] = None

    data_dir: Path = PROJECT_ROOT / "data_dir"
    collection_name: str = "abir_profile"

    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "openai/gpt-oss-20b"
    rewriter_model: str = "openai/gpt-oss-20b"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"


settings = Settings()
