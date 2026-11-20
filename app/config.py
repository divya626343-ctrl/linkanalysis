from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


   # ---- Website fetching ----
    FETCH_TIMEOUT: int = 10  # seconds per page
    MAX_RESPONSE_BYTES: int = 2_000_000  # stop reading a page after ~2 MB
    MAX_REDIRECTS: int = 3
    MAX_EXTRA_PAGES: int = 2  # about/services pages fetched besides the homepage
    MAX_CONTENT_CHARS: int = 22_000  # ~5-6k tokens of cleaned text sent to the LLM
    MIN_CONTENT_CHARS: int = 250  # below this -> INSUFFICIENT_CONTENT

   # ---- Cache ----
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_ITEMS: int = 200


settings = Settings()   