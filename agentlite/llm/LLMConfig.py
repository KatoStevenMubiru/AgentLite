import os
import unify

class LLMConfig:
    """Constructs the LLM configuration for running multi-agent system."""

    def __init__(self, config_dict: dict) -> None:
        self.config_dict = config_dict
        
        # Default settings for common LLM configurations
        self.context_len = config_dict.get("context_len", 4096)
        self.llm_name = config_dict.get("llm_name", "gpt-3.5-turbo")
        self.temperature = config_dict.get("temperature", 0.9)
        self.stop = config_dict.get("stop", ["\n"])
        self.max_tokens = config_dict.get("max_tokens", 256)
        self.end_of_prompt = config_dict.get("end_of_prompt", "")
        
        # API keys and provider specific settings
        self.api_key: str = config_dict.get("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.provider = config_dict.get("provider", "openai")
        
        # Unify-specific settings, if provider is 'unify'
        if self.provider == "unify":
            self.api_key = config_dict.get("api_key", os.environ.get("UNIFY_KEY", "EMPTY"))
            self.base_url = config_dict.get("base_url", "https://api.unify.ai/v0/")
        else:
            self.base_url = config_dict.get("base_url", None)  # Could be useful for custom endpoints
        
        self.use_custom_keys = config_dict.get("use_custom_keys", False)
        self.tags = config_dict.get("tags", [])
        
        # Validate API key presence
        if not self.api_key or self.api_key == "EMPTY":
            raise ValueError(f"API key is missing for provider {self.provider}. Please set it via environment or config.")

        # Update the rest of the fields dynamically from config_dict
        self.__dict__.update(config_dict)

    def is_unify_provider(self) -> bool:
        """Check if the provider is UnifyAI."""
        return self.provider == "unify"

    def get_llm_provider(self) -> str:
        """Returns the name of the provider (e.g., 'unify', 'openai')."""
        return self.provider

    def get_api_key(self) -> str:
        """Returns the API key for the current provider."""
        return self.api_key

    def get_llm_name(self) -> str:
        """Returns the LLM model name."""
        return self.llm_name

    def get_base_url(self) -> str:
        """Returns the base URL for the provider."""
        return self.base_url or ""
