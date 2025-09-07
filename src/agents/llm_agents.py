"""
LLM agent implementations for STT post-processing.

This module provides the core agent classes that interact with language models
to perform various text processing tasks including transcription enhancement,
entity extraction, and context analysis.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from openai import OpenAI

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.extractors import BaseResponseExtractor, JsonResponseExtractor


class BaseLLMInvoker(ABC):
    """
    Abstract base class for invoking Large Language Models.
    
    Encapsulates client setup and raw prompt-to-response logic, providing
    a consistent interface for different LLM providers and models.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", tools: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the LLM invoker.
        
        Args:
            api_key: API key for the LLM service
            model: Model identifier to use
            tools: Optional list of tools/functions for the model
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tools = tools or []

    @abstractmethod
    def invoke(self, input_text: str, extra_info: Optional[str] = None) -> str:
        """
        Run the full invocation workflow and return the final output string.
        
        Args:
            input_text: Main input text for the model
            extra_info: Additional context or information
            
        Returns:
            Model response as string
        """
        pass

    def _call_llm(self, instructions: str, prompt: str) -> str:
        """
        Low-level call to the OpenAI client.
        
        Args:
            instructions: System instructions for the model
            prompt: User prompt/input
            
        Returns:
            Raw model response
        """
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
        }
        if self.tools:
            payload["tools"] = self.tools
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()


class LLMInvoker(BaseLLMInvoker):
    """
    Simple LLM invoker that builds prompts and returns raw model output.
    
    This class provides a straightforward interface for basic LLM interactions,
    suitable for most text processing tasks in the STT pipeline.
    """

    def __init__(
        self,
        api_key: str,
        instructions: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        few_shots: Optional[List[str]] = None,
        model: str = "gpt-4o"
    ):
        """
        Initialize the LLM invoker.
        
        Args:
            api_key: API key for the LLM service
            instructions: System instructions for the model
            tools: Optional list of tools/functions
            few_shots: Few-shot examples for the model
            model: Model identifier to use
        """
        super().__init__(api_key=api_key, model=model, tools=tools)
        self.instructions = instructions
        self.few_shots = few_shots or []

    def invoke(self, input_text: str, extra_info: Optional[str] = None) -> str:
        """
        Invoke the LLM with the given input and context.
        
        Args:
            input_text: Main input text for processing
            extra_info: Additional context information
            
        Returns:
            Model response as string
        """
        parts: List[str] = []
        
        if extra_info:
            parts.append(f"Additional Info:\n{extra_info}")
        parts.append(f"Input:\n{input_text}")
        prompt = "\n\n".join(parts)

        # Prepend few-shot examples if available
        if self.few_shots:
            shots = "\n\n".join(self.few_shots)
            prompt = shots + "\n\n" + prompt

        # Call the model
        return self._call_llm(self.instructions, prompt)


class Agent(ABC):
    """
    Abstract base class for pipeline agents.
    
    Defines the interface that all agents must implement, ensuring
    consistency across the pipeline architecture.
    """

    def __init__(self, invoker: BaseLLMInvoker):
        """
        Initialize the agent with an LLM invoker.
        
        Args:
            invoker: LLM invoker instance for model communication
        """
        self.invoker = invoker

    @abstractmethod
    async def run(self, transcript: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the transcript and context to return structured data.
        
        Args:
            transcript: Input transcript text
            context: Additional context information
            
        Returns:
            Structured output data
        """
        pass


class TranscriptGenerateExtractAgent(Agent):
    """
    Agent for generating and extracting structured information from transcripts.
    
    This agent processes transcripts to extract topics, entities, and other
    contextual information that can be used to enhance STT performance.
    """

    def __init__(
        self,
        invoker: BaseLLMInvoker,
        agent_name: str,
        split_comma_to_array: bool = False,
        extract_entities_from_context: Optional[List[str]] = None,
        verbose: bool = False,
        response_extractor: Optional[BaseResponseExtractor] = None,
        return_response_with_wrapp: bool = True,
        max_retries: int = 2
    ):
        """
        Initialize the transcript extraction agent.
        
        Args:
            invoker: LLM invoker for model communication
            agent_name: Identifier for this agent
            split_comma_to_array: Whether to split comma-separated responses
            extract_entities_from_context: Context keys to extract entities from
            verbose: Enable verbose logging
            response_extractor: Custom response extraction logic
            return_response_with_wrapp: Whether to wrap responses
            max_retries: Maximum retry attempts for failed calls
        """
        super().__init__(invoker)
        self.agent_name = agent_name
        self.split_comma_to_array = split_comma_to_array
        self.extract_entities_from_context = extract_entities_from_context or []
        self.verbose = verbose
        self.response_extractor = response_extractor
        self.return_response_with_wrapp = return_response_with_wrapp
        self.max_retries = max_retries

    async def run(self, transcript: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent to process the transcript.
        
        Args:
            transcript: Input transcript text
            context: Additional context information
            
        Returns:
            Structured output with extracted information
        """
        # Build context string from relevant context keys
        context_str = ""
        if self.extract_entities_from_context:
            context_parts = []
            for key in self.extract_entities_from_context:
                if key in context:
                    context_parts.append(f"{key}: {context[key]}")
            if context_parts:
                context_str = "\n".join(context_parts)

        # Invoke the LLM
        response = self.invoker.invoke(transcript, context_str)
        
        # Extract structured data
        if self.response_extractor:
            extracted_data = self.response_extractor.extract(response)
        elif self.split_comma_to_array:
            extracted_data = [item.strip() for item in response.split(",") if item.strip()]
        else:
            extracted_data = response
        
        if self.verbose:
            print(f"[{self.agent_name}] Response: {response}")
            print(f"[{self.agent_name}] Extracted: {extracted_data}")
        
        return {
            "agent_name": self.agent_name,
            "raw_response": response,
            "extracted_data": extracted_data,
            "context_used": context_str
        }


class DeciderAgent(TranscriptGenerateExtractAgent):
    """
    Decision-making agent for determining processing strategies.
    
    This agent analyzes context and makes intelligent decisions about
    how to process transcripts based on content characteristics.
    """
    
    KEY_ANSWER = "Answer"
    DEFAULT_VALUE = "No"
    YES = "YES"

    def __init__(
        self,
        invoker: BaseLLMInvoker,
        extract_entities_from_context: Optional[List[str]] = None,
        verbose: bool = False,
        agent_name: str = "decider"
    ):
        """
        Initialize the decision agent.
        
        Args:
            invoker: LLM invoker for model communication
            extract_entities_from_context: Context keys to consider
            verbose: Enable verbose logging
            agent_name: Name of this agent
        """
        super().__init__(
            invoker=invoker,
            agent_name=agent_name,
            extract_entities_from_context=extract_entities_from_context,
            verbose=verbose,
            response_extractor=JsonResponseExtractor(key=self.KEY_ANSWER, def_val=self.DEFAULT_VALUE),
            return_response_with_wrapp=True
        )

    async def run_decision(self, transcript: str, context: Dict[str, Any]) -> bool:
        """
        Run the decision agent and return a boolean decision.
        
        Args:
            transcript: Input transcript text
            context: Additional context information
            
        Returns:
            Boolean decision (True for YES, False for NO)
        """
        result = await self.run(transcript=transcript, context=context)
        decision_value = result.get(self.agent_name, self.DEFAULT_VALUE)
        return str(decision_value).upper() == self.YES


class SentenceBuilderAgent(Agent):
    """
    Agent for building and reconstructing sentences from transcript parts.
    
    This agent takes context information and constructs coherent sentences
    or prompts that can be used for further processing.
    """

    def __init__(
        self,
        invoker: BaseLLMInvoker,
        agent_name: str
    ):
        """
        Initialize the sentence builder agent.
        
        Args:
            invoker: LLM invoker for model communication
            agent_name: Identifier for this agent
        """
        super().__init__(invoker)
        self.agent_name = agent_name

    async def run(self, context: Dict[str, Any]) -> str:
        """
        Build sentences from the provided context.
        
        Args:
            context: Context information for sentence building
            
        Returns:
            Constructed sentence or prompt
        """
        # Use context as input for sentence building
        output = self.invoker.invoke(input_text=context)
        return output
