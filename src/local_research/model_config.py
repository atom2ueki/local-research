"""Model configuration system with support for various providers including local models.

This module provides a centralized way to configure and initialize chat models
from environment variables with support for different providers:
- OpenAI: openai:gpt-4o, openai:gpt-4o-mini
- Anthropic: anthropic:claude-3-5-sonnet-20241022
- Google: google_genai:gemini-2.0-flash-exp
- LM Studio: lmstudio://localhost:1234/model-name
- Ollama: ollama://host:port/model-name

Environment variables should be prefixed with the model purpose:
- REPORT_MODEL=openai:gpt-4o
- RESEARCH_MODEL=anthropic:claude-3-5-sonnet-20241022
- SUMMARIZATION_MODEL=openai:gpt-4o-mini
- COMPRESS_MODEL=openai:gpt-4o
- SUPERVISOR_MODEL=anthropic:claude-3-5-sonnet-20241022
"""

import os
from typing import Optional, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a model string into provider and model name.

    Args:
        model_string: String in format "provider:model" or "provider://host:port/model"

    Returns:
        Tuple of (provider, model_name_or_url)

    Examples:
        "openai:gpt-4o" -> ("openai", "gpt-4o")
        "lmstudio://localhost:1234/llama-3" -> ("lmstudio", "localhost:1234/llama-3")
        "lmstudio://localhost:1234/qwen/qwen3-4b-thinking-2507" -> ("lmstudio", "localhost:1234/qwen/qwen3-4b-thinking-2507")
        "ollama://localhost:11434/gemma3-lc:12b" -> ("ollama", "localhost:11434/gemma3-lc:12b")
    """
    if "://" in model_string:
        # Handle URLs like lmstudio://localhost:1234/model or ollama://host:port/model
        parts = model_string.split("://", 1)
        provider = parts[0]
        model_info = parts[1]
        return provider, model_info
    else:
        # Handle simple format like openai:gpt-4o
        parts = model_string.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model string format: {model_string}. Expected 'provider:model' or 'provider://host:port/model'")
        return parts[0], parts[1]


def create_local_chat_model(provider: str, model_info: str, **kwargs) -> Any:
    """Create a chat model for local providers (LM Studio, Ollama).

    Args:
        provider: Provider name ("lmstudio" or "ollama")
        model_info: Model information in format "host:port/model"
        **kwargs: Additional arguments to pass to the model

    Returns:
        Configured chat model instance
    """
    # Parse host, port, and model from model_info
    if "/" not in model_info:
        raise ValueError(f"Local model info must include model name: {model_info}")

    # Find the first "/" to separate host:port from model name
    # This handles model names with "/" like "qwen/qwen3-4b-thinking-2507"
    slash_index = model_info.find("/")
    host_port = model_info[:slash_index]
    model_name = model_info[slash_index + 1:]

    if ":" not in host_port:
        raise ValueError(f"Local model info must include port: {model_info}")

    host, port = host_port.rsplit(":", 1)

    try:
        port = int(port)
    except ValueError:
        raise ValueError(f"Invalid port number: {port}")

    if provider == "lmstudio":
        # LM Studio uses OpenAI-compatible API
        return ChatOpenAI(
            base_url=f"http://{host}:{port}/v1",
            model=model_name,
            api_key="lm-studio",  # LM Studio doesn't require real API key
            **kwargs
        )
    elif provider == "ollama":
        # Import ollama package if available
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=f"http://{host}:{port}",
                model=model_name,
                **kwargs
            )
        except ImportError:
            # Fallback to OpenAI-compatible API for Ollama
            return ChatOpenAI(
                base_url=f"http://{host}:{port}/v1",
                model=model_name,
                api_key="ollama",  # Ollama doesn't require real API key
                **kwargs
            )
    else:
        raise ValueError(f"Unsupported local provider: {provider}")


def init_chat_model_from_env(env_var: str, fallback_model: str = "openai:gpt-4o", **kwargs) -> Any:
    """Initialize a chat model from environment variable with fallback.

    Args:
        env_var: Environment variable name (e.g., "REPORT_MODEL")
        fallback_model: Fallback model string if env var is not set
        **kwargs: Additional arguments to pass to init_chat_model or model constructors

    Returns:
        Configured chat model instance

    Examples:
        # Using environment variable
        os.environ["REPORT_MODEL"] = "anthropic:claude-3-5-sonnet-20241022"
        model = init_chat_model_from_env("REPORT_MODEL")

        # Using local LM Studio
        os.environ["RESEARCH_MODEL"] = "lmstudio://localhost:1234/llama-3-8b"
        model = init_chat_model_from_env("RESEARCH_MODEL")

        # Using remote Ollama
        os.environ["SUMMARIZATION_MODEL"] = "ollama://192.168.1.11:11434/gemma3-lc:12b"
        model = init_chat_model_from_env("SUMMARIZATION_MODEL")
    """
    model_string = os.getenv(env_var, fallback_model)

    try:
        provider, model_info = parse_model_string(model_string)

        # Handle local providers
        if provider in ("lmstudio", "ollama"):
            return create_local_chat_model(provider, model_info, **kwargs)

        # Handle standard providers using langchain's init_chat_model
        return init_chat_model(model=model_string, **kwargs)

    except Exception as e:
        print(f"Error initializing model from {env_var}={model_string}: {e}")
        print(f"Falling back to default model: {fallback_model}")

        # Try fallback model
        try:
            provider, model_info = parse_model_string(fallback_model)
            if provider in ("lmstudio", "ollama"):
                return create_local_chat_model(provider, model_info, **kwargs)
            return init_chat_model(model=fallback_model, **kwargs)
        except Exception as fallback_error:
            print(f"Error with fallback model {fallback_model}: {fallback_error}")
            # Final fallback to a basic OpenAI model
            return init_chat_model(model="openai:gpt-4o", **kwargs)


# Convenience functions for common model types
def get_report_model(**kwargs) -> Any:
    """Get the report generation model from REPORT_MODEL env var."""
    return init_chat_model_from_env("REPORT_MODEL", "openai:gpt-4o", **kwargs)


def get_research_model(**kwargs) -> Any:
    """Get the research model from RESEARCH_MODEL env var."""
    return init_chat_model_from_env("RESEARCH_MODEL", "anthropic:claude-3-5-sonnet-20241022", **kwargs)


def get_summarization_model(**kwargs) -> Any:
    """Get the summarization model from SUMMARIZATION_MODEL env var."""
    return init_chat_model_from_env("SUMMARIZATION_MODEL", "openai:gpt-4o-mini", **kwargs)


def get_compress_model(**kwargs) -> Any:
    """Get the compression model from COMPRESS_MODEL env var."""
    return init_chat_model_from_env("COMPRESS_MODEL", "openai:gpt-4o", **kwargs)


def get_supervisor_model(**kwargs) -> Any:
    """Get the supervisor model from SUPERVISOR_MODEL env var."""
    return init_chat_model_from_env("SUPERVISOR_MODEL", "anthropic:claude-3-5-sonnet-20241022", **kwargs)


def get_scope_model(**kwargs) -> Any:
    """Get the scoping model from SCOPE_MODEL env var."""
    return init_chat_model_from_env("SCOPE_MODEL", "openai:gpt-4o", **kwargs)
