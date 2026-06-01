"""Azure OpenAI configuration and LangChain client factories."""

import os
from urllib.parse import parse_qs, urlparse, urlunparse

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
DEFAULT_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 16


def _get_env(name, default=None):
    """Read an environment variable, tolerating whitespace around keys."""
    value = os.getenv(name)
    if value is not None:
        return value.strip()

    for key, candidate in os.environ.items():
        if key.strip() == name:
            return candidate.strip()
    return default


def _first_env(*names, default=None):
    for name in names:
        value = _get_env(name)
        if value:
            return value
    return default


def _normalize_endpoint(endpoint):
    """Return the Azure resource root and any api-version query value."""
    if not endpoint:
        return endpoint, None

    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint, None

    api_versions = parse_qs(parsed.query).get("api-version", [])
    normalized = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return normalized, api_versions[0] if api_versions else None


def _missing_settings(settings):
    return [name for name, value in settings.items() if not value]


def _get_int_env(name, default):
    value = _get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"`{name}` must be an integer.") from exc


def get_azure_openai_config(require_chat=False):
    """Return Azure OpenAI settings from environment variables."""
    endpoint = _first_env(
        "AZURE_OPENAI_EMBEDDING_ENDPOINT",
        "AZURE_OPEN_AI_ENDPOINT_EMBEDDING",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPEN_AI_ENDPOINT",
    )
    endpoint, embedding_api_version_from_endpoint = _normalize_endpoint(endpoint)
    configured_chat_endpoint = _first_env(
        "AZURE_OPENAI_CHAT_ENDPOINT",
        "AZURE_OPENAI_ENDPOINT_CHAT",
        "AZURE_OPEN_AI_ENDPOINT_CHAT",
        "AZURE_OPEN_AI_LLM_ENDPOINT",
    )
    configured_chat_endpoint, chat_api_version_from_endpoint = _normalize_endpoint(
        configured_chat_endpoint
    )
    chat_endpoint = configured_chat_endpoint or endpoint
    api_key = _first_env(
        "AZURE_OPENAI_EMBEDDING_API_KEY",
        "AZURE_OPEN_AI_KEY_EMBEDDING",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPEN_AI_KEY",
    )
    configured_chat_api_key = _first_env(
        "AZURE_OPENAI_CHAT_API_KEY",
        "AZURE_OPEN_AI_KEY_CHAT",
        "AZURE_OPEN_AI_LLM_KEY",
    )
    chat_api_key = configured_chat_api_key if configured_chat_endpoint else api_key
    api_version = _get_env(
        "AZURE_OPENAI_API_VERSION",
        embedding_api_version_from_endpoint or DEFAULT_AZURE_OPENAI_API_VERSION,
    )
    chat_api_version = _first_env(
        "AZURE_OPENAI_CHAT_API_VERSION",
        "AZURE_OPEN_AI_CHAT_API_VERSION",
        default=chat_api_version_from_endpoint or api_version,
    )
    embedding_deployment = _get_env(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        _first_env(
            "AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPEN_AI_DEPLOYMENT_EMBEDDING",
            default=DEFAULT_EMBEDDING_DEPLOYMENT,
        ),
    )
    chat_deployment = _first_env(
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPEN_AI_CHAT_DEPLOYMENT",
        "AZURE_OPEN_AI_DEPLOYMENT_CHAT",
        "AZURE_OPEN_AI_LLM_DEPLOYMENT",
    )
    chat_model = _get_env("AZURE_OPENAI_CHAT_MODEL", chat_deployment)

    required = {
        "Azure OpenAI embedding endpoint": endpoint,
        "Azure OpenAI embedding key": api_key,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": embedding_deployment,
    }
    if require_chat:
        required["Azure OpenAI chat endpoint"] = chat_endpoint
        required["Azure OpenAI chat key"] = chat_api_key
        required["AZURE_OPENAI_CHAT_DEPLOYMENT"] = chat_deployment

    missing = _missing_settings(required)
    if missing:
        raise ValueError(f"Missing Azure OpenAI environment variables: {missing}")

    return {
        "endpoint": endpoint,
        "chat_endpoint": chat_endpoint,
        "api_key": api_key,
        "chat_api_key": chat_api_key,
        "api_version": api_version,
        "chat_api_version": chat_api_version,
        "embedding_deployment": embedding_deployment,
        "chat_deployment": chat_deployment,
        "chat_model": chat_model,
    }


def get_azure_openai_embeddings(model=DEFAULT_EMBEDDING_DEPLOYMENT):
    """Create the Azure OpenAI embedding client."""
    config = get_azure_openai_config()
    deployment = model or config["embedding_deployment"]
    return AzureOpenAIEmbeddings(
        azure_endpoint=config["endpoint"],
        api_key=config["api_key"],
        api_version=config["api_version"],
        azure_deployment=deployment,
        model=deployment,
        chunk_size=_get_int_env(
            "AZURE_OPENAI_EMBEDDING_BATCH_SIZE",
            DEFAULT_EMBEDDING_BATCH_SIZE,
        ),
    )


def get_azure_chat_llm(temperature=0, model=None):
    """Create the Azure OpenAI chat client."""
    config = get_azure_openai_config(require_chat=True)
    deployment = model or config["chat_deployment"]
    return AzureChatOpenAI(
        azure_endpoint=config["chat_endpoint"],
        api_key=config["chat_api_key"],
        api_version=config["chat_api_version"],
        azure_deployment=deployment,
        model=config["chat_model"] or deployment,
        temperature=temperature,
    )
