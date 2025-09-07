"""
Enumeration definitions for STT post-processing agents and components.

This module defines the standard names and identifiers used throughout the system
for consistent agent identification and pipeline management.
"""

from enum import Enum


class AgentNames(Enum):
    """
    Standard names for LLM agents in the STT post-processing pipeline.
    
    These names are used to identify agents and maintain consistency across
    different pipeline implementations and evaluation metrics.
    """
    
    # Core extraction agents
    TOPIC = "topic"
    NER_NAMES = "names"
    JRAGON_LIST = "jargon_list"
    
    # Decision-making agents
    NER_DECIDER = "decider_ner"
    JARGON_DEIDER = "decider_jargon"
    
    # Specialized processing agents
    MOST_RELEVANT_NAMES = "most_relevant_names"
    SENTENCE_BUILDER = "sentence_builder"
    FIX_TRANSCRIPT_BY_LLM = "fix_transcript_by_llm"
    
    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value
    
    @classmethod
    def get_friendly_name(cls, agent_name: str) -> str:
        """
        Convert agent name to human-readable format.
        
        Args:
            agent_name: Raw agent name string
            
        Returns:
            Human-readable agent name
            
        Example:
            >>> AgentNames.get_friendly_name("topic")
            'Topic Analysis'
        """
        friendly_names = {
            "topic": "Topic Analysis",
            "names": "Named Entity Recognition",
            "jargon_list": "Jargon Extraction",
            "decider_ner": "NER Decision Agent",
            "decider_jargon": "Jargon Decision Agent",
            "most_relevant_names": "Relevant Names Filter",
            "sentence_builder": "Sentence Builder",
            "fix_transcript_by_llm": "Transcript Correction",
        }
        return friendly_names.get(agent_name, agent_name.title())
