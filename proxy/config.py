from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BACKEND_URL: str = "http://backend:8000"

    class Config:
        env_file = ".env"


settings = Settings()
