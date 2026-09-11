"""
Centralized configuration. All values come from environment variables
(.env for local dev) — never hardcode secrets here.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Document Intelligence Platform"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "sqlite:///./data/app.db"

    max_pages: int = 3
    max_file_size_mb: int = 10
    upload_dir: str = "./data/uploads"

    tesseract_cmd: str | None = None

    # Deterministic OCR extraction is the fast/default path. Set USE_LLM_FALLBACK=true
    # only when an OCR parser cannot recover enough fields.
    use_llm_fallback: bool = True
    llm_provider: str = "huggingface"  # huggingface | openai | gemini | anthropic
    huggingface_api_key: str | None = None
    huggingface_model: str = "deepseek-ai/DeepSeek-R1"
    huggingface_base_url: str = "https://router.huggingface.co/hf-inference/v1"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")



settings = Settings()
