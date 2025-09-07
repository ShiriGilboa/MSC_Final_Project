"""
Configuration file for STT Evaluation Project.
Contains API keys, model settings, and other configuration parameters.
"""

import os
from typing import Optional

# OpenAI Configuration
OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')

# Model Configuration
DEFAULT_MODEL: str = "gpt-4"
MAX_TOKENS: int = 1000
TEMPERATURE: float = 0.1

# Streamlit Dashboard Configuration
DASHBOARD_PASSWORD: str = os.getenv('STREAMLIT_PASSWORD', 'Yonathan_Shiri_2025')

# Dataset Configuration
HUGGINGFACE_DATASET: str = "ShiriGilboa/my-nba-dataset"

# Pipeline Configuration
DEFAULT_ITERATIONS: int = 1
VERBOSE_LOGGING: bool = True

# File Paths
DATA_DIR: str = "data"
OUTPUT_DIR: str = "output"
RESULTS_FILE: str = "evaluation_results_unified.csv"

def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from environment or config.
    
    Returns:
        The OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.\n"
            "You can do this by running:\n"
            "export OPENAI_API_KEY='your-api-key-here'\n"
            "Or add it to your ~/.zshrc file for persistence."
        )
    return OPENAI_API_KEY

def validate_config() -> bool:
    """
    Validate that all required configuration is present.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        get_openai_api_key()
        return True
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return False
