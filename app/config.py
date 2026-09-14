"""Central app configuration.

All settings are read from environment variables (populated from `.env` in
dev, and from AWS App Runner environment variables in production). Nothing
here should ever hold a real secret value — see `.env.example` for the full
list of variable names.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Secrets default to "" rather than being required: this lets the app
    # boot (and /health respond, and CI run tests that don't touch these
    # services) without real keys present. Calls that actually need a key
    # fail obviously and immediately at the client, not silently here.

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-app-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # OpenAI — embeddings and answer generation both go through OpenAI, via
    # LangChain. text-embedding-3-small is 1536-dim; the Pinecone index
    # dimension in app/vectorstore.py must match.
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # App
    app_env: str = "dev"
    port: int = 8000


# Import `settings` from this module rather than instantiating Settings()
# directly elsewhere, so the whole app shares one validated config object.
settings = Settings()
