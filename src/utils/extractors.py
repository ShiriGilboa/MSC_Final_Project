"""
Response extraction utilities for LLM outputs.

This module provides strategies for extracting structured data from raw LLM responses,
including JSON parsing and key-based extraction with robust error handling.
"""

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Optional


class BaseResponseExtractor(ABC):
    """
    Abstract base class for response extraction strategies.
    
    Defines the interface for all response extractors, allowing different
    parsing strategies to be used interchangeably in the pipeline.
    """

    @abstractmethod
    def extract(self, response: str) -> Any:
        """
        Extract structured data from the raw LLM response string.

        Args:
            response: Raw text output from the LLM.

        Returns:
            Parsed/processed data in a structured form.
            
        Raises:
            ValueError: If the response cannot be parsed
        """
        pass


class JsonResponseExtractor(BaseResponseExtractor):
    """
    Extractor that parses LLM responses as JSON with optional key extraction.
    
    This extractor handles common LLM response formatting issues and provides
    robust JSON parsing with support for nested key access.
    """

    def __init__(self, key: Optional[str] = None, def_val: str = ""):
        """
        Initialize the JSON extractor.
        
        Args:
            key: Optional key to extract from the parsed JSON
            def_val: Default value if key extraction fails
        """
        self.key = key
        self.def_val = def_val

    def _clean_response(self, response: str) -> str:
        """
        Clean the LLM response to extract valid JSON.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Cleaned string containing valid JSON
        """
        # Remove markdown fences and language hints
        cleaned = re.sub(r"```(?:json)?", "", response, flags=re.IGNORECASE)
        
        # Trim stray backticks and whitespace
        cleaned = cleaned.strip("`\n ")
        
        # Extract only the first valid JSON object
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return match.group(0)
        return cleaned

    def extract(self, response: str) -> Any:
        """
        Extract structured data from the LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed JSON data or extracted key value
            
        Raises:
            ValueError: If JSON parsing fails
            KeyError: If the specified key is not found
        """
        cleaned = self._clean_response(response)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse JSON:\n{e}\n\n"
                f"Cleaned response was:\n{cleaned!r}"
            )

        if self.key is None:
            return data

        # Support nested keys via dot notation
        current = data
        for part in self.key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Return default value if key not found
                if self.def_val:
                    return self.def_val
                raise KeyError(f"Key '{self.key}' not found in JSON response")
        
        return current


class TextResponseExtractor(BaseResponseExtractor):
    """
    Simple text extractor that returns cleaned text responses.
    
    Useful for cases where the LLM response is plain text that doesn't
    require structured parsing.
    """

    def __init__(self, remove_markdown: bool = True):
        """
        Initialize the text extractor.
        
        Args:
            remove_markdown: Whether to remove markdown formatting
        """
        self.remove_markdown = remove_markdown

    def extract(self, response: str) -> str:
        """
        Extract clean text from the LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Cleaned text string
        """
        if not self.remove_markdown:
            return response.strip()
        
        # Remove markdown formatting
        cleaned = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
        cleaned = re.sub(r"`.*?`", "", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
        
        return cleaned.strip()


class ListResponseExtractor(BaseResponseExtractor):
    """
    Extractor for responses that should be parsed as lists.
    
    Handles various list formats including comma-separated values,
    bullet points, and numbered lists.
    """

    def __init__(self, separator: str = ",", remove_bullets: bool = True):
        """
        Initialize the list extractor.
        
        Args:
            separator: Character to split on for comma-separated lists
            remove_bullets: Whether to remove bullet points and numbers
        """
        self.separator = separator
        self.remove_bullets = remove_bullets

    def extract(self, response: str) -> list:
        """
        Extract list data from the LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            List of extracted items
        """
        cleaned = response.strip()
        
        if self.remove_bullets:
            # Remove bullet points and numbers
            cleaned = re.sub(r"^[\s]*[-*•]\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"^[\s]*\d+\.\s*", "", cleaned, flags=re.MULTILINE)
        
        # Split by separator and clean each item
        items = [item.strip() for item in cleaned.split(self.separator)]
        
        # Remove empty items
        items = [item for item in items if item]
        
        return items
