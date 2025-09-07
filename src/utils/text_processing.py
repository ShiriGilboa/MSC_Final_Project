"""
Text processing utilities for STT post-processing evaluation.

This module provides functions for text normalization, YAML parsing, and prompt building
that are essential for consistent evaluation of STT outputs.
"""

import re
import yaml
import num2words
from typing import List, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent evaluation by converting numbers to words,
    removing special characters, and standardizing formatting.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalize_text("Hello123 world!")
        'hello one hundred and twenty three world'
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    def replace_number(match):
        """Convert matched number to word representation."""
        try:
            return num2words.num2words(int(match.group()))
        except Exception:
            return match.group()

    # Replace numbers with words before removing special characters
    text = re.sub(r'\d+', replace_number, text)
    
    # Remove apostrophes to match baseline output format
    text = text.replace("'", "")
    
    # Remove other special characters but keep letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def parse_yaml_block(text: str) -> Tuple[str, List[str]]:
    """
    Extract YAML block from text and return description and terms.
    
    Args:
        text: Text containing YAML block
        
    Returns:
        Tuple of (description, list_of_terms)
        
    Example:
        >>> text = "```yaml\nshort_context_description: Technical discussion\nlist_of_terms: [AI, ML, NLP]\n```"
        >>> parse_yaml_block(text)
        ('Technical discussion', ['AI', 'ML', 'NLP'])
    """
    pattern = r"```yaml\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return "", []

    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        print(f"[WARN] YAML parse error: {exc}")
        return "", []

    desc = (data.get("short_context_description") or "").strip()

    # Extract terms from various possible keys
    terms = (
        data.get("nlist_of_terms")
        or data.get("list_of_terms")
        or data.get("list_of_unique_terms")
        or []
    )
    
    # Handle comma-separated string format
    if isinstance(terms, str):
        terms = [s.strip() for s in re.split(r",\s*", terms) if s.strip()]

    return desc, terms


def build_plain_prompt(desc: str, terms: List[str]) -> str:
    """
    Convert description and terms into Whisper-friendly plain text prompt.
    
    Args:
        desc: Context description
        terms: List of relevant terms
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> build_plain_prompt("Technical discussion", ["AI", "ML"])
        'Technical discussion. It's AI, ML.'
    """
    glossary = ", ".join(terms)
    if desc and glossary:
        return f"{desc}. It's {glossary}."
    return desc or glossary
