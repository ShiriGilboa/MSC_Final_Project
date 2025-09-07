"""
Configuration file for STT Post-Processing Evaluation Project.

This file contains all the configurable parameters and settings
for the project, making it easy to customize behavior without
modifying the core code.
"""

import os
from typing import Dict, Any

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Dashboard Configuration
DASHBOARD_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "stt_dashboard_2024")
DASHBOARD_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Pipeline Configuration
PIPELINE_CONFIG = {
    "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
    "verbose_logging": os.getenv("VERBOSE_LOGGING", "false").lower() == "true",
    "timeout_seconds": int(os.getenv("TIMEOUT_SECONDS", "30")),
}

# Text Processing Configuration
TEXT_PROCESSING_CONFIG = {
    "normalize_numbers": True,
    "remove_apostrophes": True,
    "remove_special_chars": True,
    "min_word_length": 2,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "wer_threshold": float(os.getenv("WER_THRESHOLD", "0.1")),
    "improvement_threshold": float(os.getenv("IMPROVEMENT_THRESHOLD", "0.05")),
    "min_segments_for_analysis": int(os.getenv("MIN_SEGMENTS", "10")),
}

# File Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM Agent Instructions (can be customized)
AGENT_INSTRUCTIONS = {
    "topic_extraction": {
        "max_words": 5,
        "style": "concise",
        "focus": "domain_identification"
    },
    "ner_extraction": {
        "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"],
        "similarity_threshold": 0.85,
        "normalization": "reference_based"
    },
    "jargon_extraction": {
        "extraction_methods": ["TF-IDF", "RAKE", "YAKE"],
        "similarity_threshold": 0.90,
        "exclude_names": True
    }
}

# Pipeline Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "min_improvement_rate": 0.1,  # 10%
    "max_processing_time": 30,    # seconds
    "min_confidence_score": 0.7,  # 70%
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(LOGS_DIR, "stt_evaluation.log"),
    "max_file_size": "10MB",
    "backup_count": 5
}

# Dashboard Theme Configuration
DASHBOARD_THEME = {
    "primary_color": "#1f77b4",
    "secondary_color": "#ff7f0e",
    "background_color": "#ffffff",
    "text_color": "#333333",
    "accent_color": "#2ca02c"
}

# Export Configuration
EXPORT_CONFIG = {
    "formats": ["csv", "json", "excel"],
    "include_metadata": True,
    "include_timestamps": True,
    "compression": "gzip"
}

# Validation Configuration
VALIDATION_CONFIG = {
    "validate_inputs": True,
    "validate_outputs": True,
    "strict_mode": False,
    "error_handling": "graceful"  # or "strict"
}


def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    return {
        "api": {
            "openai_key": OPENAI_API_KEY,
            "openai_model": OPENAI_MODEL
        },
        "dashboard": {
            "password": DASHBOARD_PASSWORD,
            "port": DASHBOARD_PORT,
            "theme": DASHBOARD_THEME
        },
        "pipeline": PIPELINE_CONFIG,
        "text_processing": TEXT_PROCESSING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "paths": {
            "data": DATA_DIR,
            "results": RESULTS_DIR,
            "logs": LOGS_DIR
        },
        "agents": AGENT_INSTRUCTIONS,
        "performance": PERFORMANCE_THRESHOLDS,
        "logging": LOGGING_CONFIG,
        "export": EXPORT_CONFIG,
        "validation": VALIDATION_CONFIG
    }


def validate_config() -> bool:
    """
    Validate the configuration parameters.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")
    
    if PIPELINE_CONFIG["max_iterations"] < 1:
        errors.append("MAX_ITERATIONS must be at least 1")
    
    if EVALUATION_CONFIG["wer_threshold"] < 0 or EVALUATION_CONFIG["wer_threshold"] > 1:
        errors.append("WER_THRESHOLD must be between 0 and 1")
    
    if EVALUATION_CONFIG["improvement_threshold"] < 0:
        errors.append("IMPROVEMENT_THRESHOLD must be non-negative")
    
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True


def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    config = get_config()
    
    print("🔧 STT Evaluation Project Configuration Summary")
    print("=" * 50)
    
    print(f"📊 API Model: {config['api']['openai_model']}")
    print(f"🔐 Dashboard Port: {config['dashboard']['port']}")
    print(f"⚙️ Max Iterations: {config['pipeline']['max_iterations']}")
    print(f"📁 Data Directory: {config['paths']['data']}")
    print(f"📁 Results Directory: {config['paths']['results']}")
    print(f"📁 Logs Directory: {config['paths']['logs']}")
    
    print(f"\n📈 Performance Thresholds:")
    print(f"  - Min Improvement Rate: {config['performance']['min_improvement_rate']*100}%")
    print(f"  - Max Processing Time: {config['performance']['max_processing_time']}s")
    print(f"  - Min Confidence Score: {config['performance']['min_confidence_score']*100}%")
    
    print(f"\n🔍 Validation:")
    print(f"  - Input Validation: {config['validation']['validate_inputs']}")
    print(f"  - Output Validation: {config['validation']['validate_outputs']}")
    print(f"  - Strict Mode: {config['validation']['strict_mode']}")


if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()
    
    # Validate configuration
    print("\n" + "=" * 50)
    validate_config()
