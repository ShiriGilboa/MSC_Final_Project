"""
Core pipeline implementations for STT post-processing.

This module provides the main pipeline classes that orchestrate LLM agents
to perform various text processing tasks including transcription enhancement,
entity extraction, and context analysis.
"""

import asyncio
from abc import ABC
from typing import Any, Dict, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.llm_agents import LLMInvoker, DeciderAgent, SentenceBuilderAgent, TranscriptGenerateExtractAgent
from utils.enums import AgentNames
from utils.extractors import JsonResponseExtractor
from utils.instructions import (
    BEST_CANDIDATES_AGENT_INSTRUCTIONS,
    BUILD_SENTENCE_FROM_PARTS,
    JARGON_AGENT_INSTRUCTIONS,
    JARGON_DECIDER_AGENT_INSTRUCTIONS,
    NER_AGENT_INSTRUCTIONS,
    NER_DECIDER_AGENT_INSTRUCTIONS,
    TOPIC_INSTRUCTIONS,
    FIX_STT_OUTPUT_AGENT_INSTRUCTIONS
)


class BasePipeline(ABC):
    """
    Abstract base class for all processing pipelines.
    
    Defines the common interface and functionality that all pipelines
    must implement, ensuring consistency across the system.
    """

    def __init__(self, api_key: str, pipeline_name: str = "BasePipeline", 
                 num_iterations_allowed: int = 1, verbose: bool = False):
        """
        Initialize the base pipeline.
        
        Args:
            api_key: API key for LLM services
            pipeline_name: Human-readable name for this pipeline
            num_iterations_allowed: Maximum number of processing iterations
            verbose: Enable verbose logging
        """
        self.api_key = api_key
        self.pipeline_name = pipeline_name
        self.num_iterations_allowed = num_iterations_allowed
        self.verbose = verbose

    def get_num_iterations_allowed(self) -> int:
        """Return the number of iterations allowed for this pipeline."""
        return self.num_iterations_allowed

    def get_pipeline_name(self) -> str:
        """Return the human-readable name of this pipeline."""
        return self.pipeline_name

    async def process(self, transcript: str) -> Tuple[bool, str]:
        """
        Process the transcript and return success status and result.
        
        Args:
            transcript: The input text to process.

        Returns:
            Tuple of (is_generated: bool, result: str).
        """
        raise NotImplementedError("Subclasses must implement this method.")


class GenerateWhisperPromptPipeline(BasePipeline):
    """
    Pipeline for generating context-aware prompts for Whisper STT.
    
    This pipeline analyzes transcripts to extract topics, entities, and jargon,
    then generates optimized prompts that can improve Whisper's transcription accuracy.
    """

    def __init__(
        self,
        api_key: str,
        pipeline_name: str = "GenerateWhisperPromptPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the Whisper prompt generation pipeline.
        
        Args:
            api_key: API key for LLM services
            pipeline_name: Name for this pipeline instance
            num_iterations_allowed: Maximum processing iterations
            verbose: Enable verbose logging
        """
        super().__init__(api_key, pipeline_name, num_iterations_allowed, verbose)

        # Initialize specialized agents
        self.topic_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(api_key, instructions=TOPIC_INSTRUCTIONS),
            verbose=verbose,
            agent_name=AgentNames.TOPIC.value
        )
        
        self.ner_agent = TranscriptGenerateExtractAgent(
            LLMInvoker(api_key, instructions=NER_AGENT_INSTRUCTIONS),
            split_comma_to_array=True,
            extract_entities_from_context=[AgentNames.TOPIC.value],
            verbose=verbose,
            agent_name=AgentNames.NER_NAMES.value
        )
        
        self.jargon_agent = TranscriptGenerateExtractAgent(
            LLMInvoker(api_key, instructions=JARGON_AGENT_INSTRUCTIONS),
            split_comma_to_array=True,
            extract_entities_from_context=[AgentNames.TOPIC.value],
            verbose=verbose,
            agent_name=AgentNames.JRAGON_LIST.value
        )
        
        self.decider_ner = DeciderAgent(
            LLMInvoker(api_key, instructions=NER_DECIDER_AGENT_INSTRUCTIONS),
            extract_entities_from_context=[AgentNames.TOPIC.value, AgentNames.NER_NAMES.value],
            verbose=verbose,
            agent_name=AgentNames.NER_DECIDER.value
        )
        
        self.decider_jargon = DeciderAgent(
            LLMInvoker(api_key, instructions=JARGON_DECIDER_AGENT_INSTRUCTIONS),
            extract_entities_from_context=[AgentNames.TOPIC.value, AgentNames.JRAGON_LIST.value],
            verbose=verbose,
            agent_name=AgentNames.JARGON_DEIDER.value
        )
        
        self.choose_relevant_names = TranscriptGenerateExtractAgent(
            LLMInvoker(api_key, instructions=BEST_CANDIDATES_AGENT_INSTRUCTIONS),
            extract_entities_from_context=[AgentNames.TOPIC.value, AgentNames.NER_NAMES.value],
            verbose=verbose,
            agent_name=AgentNames.MOST_RELEVANT_NAMES.value,
            response_extractor=JsonResponseExtractor(key=AgentNames.NER_NAMES.value, def_val=""),
            return_response_with_wrapp=False
        )
        
        self.sentence_builder = SentenceBuilderAgent(
            LLMInvoker(api_key, instructions=BUILD_SENTENCE_FROM_PARTS),
            agent_name=AgentNames.SENTENCE_BUILDER.value
        )

    async def process(self, transcript: str) -> Tuple[bool, str, str]:
        """
        Process transcript to generate Whisper-optimized prompts.
        
        Args:
            transcript: Input transcript text
            
        Returns:
            Tuple of (success, prompt, context_info)
        """
        ret_val: str = ""
        is_generated_initial_prompt: bool = False
        ctx: Dict[str, Any] = {}

        # Step 1: Extract topic
        topic_res = await self.topic_agent.run(transcript, ctx)
        ctx.update(topic_res)
        
        # Step 2: Extract NER and jargon in parallel
        ner_res, jargon_res = await asyncio.gather(
            self.ner_agent.run(transcript, ctx),
            self.jargon_agent.run(transcript, ctx)
        )
        ctx.update(ner_res)
        ctx.update(jargon_res)

        # Step 3: Decide whether to include names and jargon terms
        ner_decider_decision, jargon_decider_decision = await asyncio.gather(
            self.decider_ner.run(transcript, ctx),
            self.decider_jargon.run(transcript, ctx)
        )
          
        if ner_decider_decision and ner_decider_decision.get(AgentNames.NER_DECIDER.value, "").upper() == "YES":
            best_names = await self.choose_relevant_names.run(transcript, ctx)
            print(f"Best Names: {best_names}")
            sent_ctx: Dict[str, Any] = {}
            sent_ctx[AgentNames.TOPIC] = ctx.get(AgentNames.TOPIC, "")
            sent_ctx["names_list"] = best_names

            if jargon_decider_decision and jargon_decider_decision.get(AgentNames.JARGON_DEIDER.value, "").upper() == "YES":
                sent_ctx[AgentNames.JRAGON_LIST] = ctx.get(AgentNames.JRAGON_LIST, "")

            res = await self.sentence_builder.run(sent_ctx)
            print(f"sentence build: {res}")
            ret_val = res
            is_generated_initial_prompt = True

        return is_generated_initial_prompt, ret_val


class FixTranscriptByLLMPipeline(BasePipeline):
    """
    Pipeline for post-processing and correcting STT transcripts using LLMs.
    
    This pipeline takes raw STT output and applies intelligent corrections
    to improve accuracy, grammar, and contextual understanding.
    """

    def __init__(
        self,
        api_key: str,
        pipeline_name: str = "FixTranscriptByLLMPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False
    ):
        """Initialize the transcript fixing pipeline."""
        super().__init__(api_key, pipeline_name, num_iterations_allowed, verbose)
        
        self.stt_output_corrector_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(api_key, instructions=FIX_STT_OUTPUT_AGENT_INSTRUCTIONS),
            verbose=verbose,
            agent_name=AgentNames.FIX_TRANSCRIPT_BY_LLM.value,
            return_response_with_wrapp=False
        )

    async def process(self, transcript: str) -> Tuple[bool, str]:
        """
        Fix and improve the transcript using LLM analysis.
        
        Args:
            transcript: Raw transcript text to fix
            
        Returns:
            Tuple of (success, corrected_transcript)
        """
        ret_val: str = ""
        ctx: Dict[str, Any] = {}
        
        ret_val = await self.stt_output_corrector_agent.run(transcript, ctx)
        return False, ret_val


class GenerateNamesPipeline(BasePipeline):
    """
    Pipeline for extracting and processing named entities from transcripts.
    
    Focuses specifically on identifying and categorizing proper nouns,
    names, and entities mentioned in the conversation.
    """

    def __init__(
        self,
        api_key: str,
        pipeline_name: str = "GenerateNamesPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False
    ):
        """Initialize the names generation pipeline."""
        super().__init__(api_key, pipeline_name, num_iterations_allowed, verbose)
        
        self.topic_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(api_key, instructions=TOPIC_INSTRUCTIONS),
            verbose=verbose,
            agent_name=AgentNames.TOPIC.value
        )
        
        self.ner_agent = TranscriptGenerateExtractAgent(
            LLMInvoker(api_key, instructions=NER_AGENT_INSTRUCTIONS),
            split_comma_to_array=True,
            extract_entities_from_context=[AgentNames.TOPIC.value],
            verbose=verbose,
            agent_name=AgentNames.NER_NAMES.value
        )

    async def process(self, transcript: str) -> Tuple[bool, str]:
        """
        Extract named entities from the transcript.
        
        Args:
            transcript: Input transcript text
            
        Returns:
            Tuple of (success, comma_separated_names)
        """
        ret_val: str = ""
        ctx: Dict[str, Any] = {}
        
        topic_res = await self.topic_agent.run(transcript, ctx)
        ctx.update(topic_res)
        ner_res = await self.ner_agent.run(transcript, ctx)
        ret_val = ', '.join(ner_res.get("extracted_data", []))
        if self.verbose:
            print(f"Extracted Names: {ret_val}")
        return True, ret_val


class GenerateTopicPipeline(BasePipeline):
    """
    Pipeline for identifying and categorizing conversation topics.
    
    Analyzes transcript content to determine the main subject matter,
    context, and thematic elements of the conversation.
    """

    def __init__(
        self,
        api_key: str,
        pipeline_name: str = "GenerateTopicPipeline",
        num_iterations_allowed: int = 1,
        verbose: bool = False
    ):
        """Initialize the topic generation pipeline."""
        super().__init__(api_key, pipeline_name, num_iterations_allowed, verbose)
        
        self.topic_agent = TranscriptGenerateExtractAgent(
            invoker=LLMInvoker(api_key, instructions=TOPIC_INSTRUCTIONS),
            verbose=verbose,
            agent_name=AgentNames.TOPIC.value
        )

    async def process(self, transcript: str) -> Tuple[bool, str]:
        """
        Extract the main topic from the transcript.
        
        Args:
            transcript: Input transcript text
            
        Returns:
            Tuple of (success, topic_description)
        """
        ctx: Dict[str, Any] = {}
        
        ret_val = await self.topic_agent.run(transcript, ctx)
        return True, ret_val.get("extracted_data", "")
