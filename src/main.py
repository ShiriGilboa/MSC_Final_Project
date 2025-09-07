#!/usr/bin/env python3
"""
Main entry point for STT Post-Processing Evaluation Project.

This script provides a command-line interface for running various pipelines
and can be used as an alternative to Jupyter notebooks for reproducible execution.
"""

import asyncio
import argparse
import sys
import os
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import real pipelines
try:
    from stt_evaluation.core.pipelines import (
        GenerateWhisperPromptPipeline,
        FixTranscriptByLLMPipeline,
        GenerateNamesPipeline,
        GenerateTopicPipeline
    )
    REAL_PIPELINES_AVAILABLE = True
except ImportError:
    REAL_PIPELINES_AVAILABLE = False
    print("❌ Error: Could not import pipeline modules.")
    print("💡 Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)


async def run_pipeline(pipeline_name: str, transcript: str, api_key: str, verbose: bool = False) -> None:
    """
    Run a specific pipeline with the given transcript.
    
    Args:
        pipeline_name: Name of the pipeline to run
        transcript: Input transcript text
        api_key: OpenAI API key
        verbose: Enable verbose logging
    """
    print(f"🚀 Running {pipeline_name}...")
    print(f"📝 Input transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    
    try:
        if pipeline_name == "whisper_prompt":
            pipeline = GenerateWhisperPromptPipeline(api_key, verbose=verbose)
            success, result, context = await pipeline.process(transcript)
            print(f"✅ Success: {success}")
            print(f"📤 Result: {result}")
            print(f"🔍 Context: {context}")
            
        elif pipeline_name == "fix_transcript":
            pipeline = FixTranscriptByLLMPipeline(api_key, verbose=verbose)
            success, result = await pipeline.process(transcript)
            print(f"✅ Success: {success}")
            print(f"📤 Corrected transcript: {result}")
            
        elif pipeline_name == "extract_names":
            pipeline = GenerateNamesPipeline(api_key, verbose=verbose)
            success, result = await pipeline.process(transcript)
            print(f"✅ Success: {success}")
            print(f"📤 Extracted names: {result}")
            
        elif pipeline_name == "extract_topic":
            pipeline = GenerateTopicPipeline(api_key, verbose=verbose)
            success, result = await pipeline.process(transcript)
            print(f"✅ Success: {success}")
            print(f"📤 Extracted topic: {result}")
            
        else:
            print(f"❌ Unknown pipeline: {pipeline_name}")
            return
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


async def run_batch_evaluation(api_key: str, input_file: str, output_file: str, verbose: bool = False) -> None:
    """
    Run batch evaluation on multiple transcripts from a file.
    
    Args:
        api_key: OpenAI API key
        input_file: Path to input file with transcripts
        output_file: Path to output file for results
        verbose: Enable verbose logging
    """
    print(f"📁 Running batch evaluation...")
    print(f"📥 Input file: {input_file}")
    print(f"📤 Output file: {output_file}")
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            transcripts = [line.strip() for line in f if line.strip()]
        
        print(f"📊 Found {len(transcripts)} transcripts to process")
        
        # Process each transcript
        results = []
        for i, transcript in enumerate(transcripts, 1):
            print(f"\n🔄 Processing transcript {i}/{len(transcripts)}")
            
            # Run all pipelines
            pipeline_results = {}
            
            # Whisper prompt pipeline
            pipeline = GenerateWhisperPromptPipeline(api_key, verbose=verbose)
            success, result, context = await pipeline.process(transcript)
            pipeline_results['whisper_prompt'] = {
                'success': success,
                'result': result,
                'context': context
            }
            
            # Fix transcript pipeline
            pipeline = FixTranscriptByLLMPipeline(api_key, verbose=verbose)
            success, result = await pipeline.process(transcript)
            pipeline_results['fix_transcript'] = {
                'success': success,
                'result': result
            }
            
            # Extract names pipeline
            pipeline = GenerateNamesPipeline(api_key, verbose=verbose)
            success, result = await pipeline.process(transcript)
            pipeline_results['extract_names'] = {
                'success': success,
                'result': result
            }
            
            # Extract topic pipeline
            pipeline = GenerateTopicPipeline(api_key, verbose=verbose)
            success, result = await pipeline.process(transcript)
            pipeline_results['extract_topic'] = {
                'success': success,
                'result': result
            }
            
            results.append({
                'transcript_id': i,
                'original_transcript': transcript,
                'pipeline_results': pipeline_results
            })
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Batch evaluation completed! Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error in batch evaluation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="STT Post-Processing Evaluation Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single pipeline
  python src/main.py --pipeline whisper_prompt --transcript "Hello world" --api-key YOUR_KEY
  
  # Run batch evaluation
  python src/main.py --batch --input transcripts.txt --output results.json --api-key YOUR_KEY
  
  # Run with verbose logging
  python src/main.py --pipeline fix_transcript --transcript "Sample text" --api-key YOUR_KEY --verbose
        """
    )
    
    # Pipeline selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--pipeline',
        choices=['whisper_prompt', 'fix_transcript', 'extract_names', 'extract_topic'],
        help='Pipeline to run'
    )
    group.add_argument(
        '--batch',
        action='store_true',
        help='Run batch evaluation on multiple transcripts'
    )
    
    # Input/output arguments
    parser.add_argument(
        '--transcript',
        help='Input transcript text (for single pipeline mode)'
    )
    parser.add_argument(
        '--input',
        help='Input file path (for batch mode)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (for batch mode)'
    )
    
    # Configuration
    parser.add_argument(
        '--api-key',
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pipeline and not args.transcript:
        parser.error("--transcript is required when using --pipeline")
    
    if args.batch and (not args.input or not args.output):
        parser.error("--input and --output are required when using --batch")
    
    # Check if API key is provided
    if not args.api_key:
        parser.error("--api-key is required for pipeline execution")
    
    # Run the appropriate mode
    if args.batch:
        asyncio.run(run_batch_evaluation(args.api_key, args.input, args.output, args.verbose))
    else:
        asyncio.run(run_pipeline(args.pipeline, args.transcript, args.api_key, args.verbose))


if __name__ == "__main__":
    main()
