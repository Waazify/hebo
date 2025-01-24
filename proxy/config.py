from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Environment settings
    TARGET_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    # Backend settings
    BACKEND_URL: str = "http://backend:8000"

    # Database settings
    DB_NAME: str | None = None
    DB_USER: str | None = None
    DB_PASS: str | None = None
    DB_PORT: str | None = None
    DB_HOST: str | None = None
    # Langfuse settings
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_HOST: str | None = None

    # Model keys
    VOYAGE_API_KEY: str | None = None

    # General settings
    ARTIFICIAL_DELAY_DURATION: int = 10
    MAX_RECURSION_DEPTH: int = 5
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
