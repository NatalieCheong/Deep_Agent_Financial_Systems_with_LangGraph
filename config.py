"""
Configuration for DeepAgent Financial Systems
Based on LangChain deep-agents-from-scratch patterns
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for DeepAgent Financial Systems"""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    
    # LangSmith Configuration
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "deepagent-financial-systems")
    
    # Model Configuration
    DEFAULT_MODEL: str = "gpt-4o-mini"
    REASONING_MODEL: str = "gpt-4o"
    FAST_MODEL: str = "gpt-3.5-turbo"
    
    # Financial Data Configuration
    DEFAULT_MARKET: str = os.getenv("DEFAULT_MARKET", "US")
    DEFAULT_CURRENCY: str = os.getenv("DEFAULT_CURRENCY", "USD")
    MAX_HISTORICAL_DAYS: int = int(os.getenv("MAX_HISTORICAL_DAYS", "365"))
    
    # Agent Configuration
    MAX_ITERATIONS: int = 50
    MAX_EXECUTION_TIME: int = 300  # seconds
    
    # File System Configuration (for virtual file system)
    MAX_FILES: int = 100
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "tavily_configured": bool(cls.TAVILY_API_KEY),
            "langsmith_configured": bool(cls.LANGSMITH_API_KEY),
            "tracing_enabled": cls.LANGSMITH_TRACING,
        }
        return status
    
    @classmethod
    def get_model_settings(cls, model_type: str = "default") -> Dict[str, Any]:
        """Get model settings for different use cases"""
        models = {
            "default": {
                "model": cls.DEFAULT_MODEL,
                "temperature": 0.1,
                "max_tokens": 4000,
            },
            "reasoning": {
                "model": cls.REASONING_MODEL,
                "temperature": 0.0,
                "max_tokens": 8000,
            },
            "fast": {
                "model": cls.FAST_MODEL,
                "temperature": 0.2,
                "max_tokens": 2000,
            },
            "creative": {
                "model": cls.DEFAULT_MODEL,
                "temperature": 0.7,
                "max_tokens": 4000,
            }
        }
        return models.get(model_type, models["default"])

# Global configuration instance
config = Config()

# Validate configuration on import
if __name__ == "__main__":
    status = config.validate_config()
    print("Configuration Status:")
    for key, value in status.items():
        print(f"  {key}: {'✓' if value else '✗'}")