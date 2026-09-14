"""Central app configuration.

All settings are read from environment variables (populated from `.env` in
dev, and from AWS App Runner environment variables in production). Nothing
here should ever hold a real secret value — see `.env.example` for the full
list of variable names.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "rag-app-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # HuggingFace Inference API
    hf_api_token: str
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Groq
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # App
    app_env: str = "dev"
    port: int = 8000


# Import `settings` from this module rather than instantiating Settings()
# directly elsewhere, so the whole app shares one validated config object.
settings = Settings()
