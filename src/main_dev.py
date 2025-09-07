#!/usr/bin/env python3
"""
Development version of main entry point for STT Post-Processing Evaluation Project.

This script includes demo mode functionality for testing and development purposes.
NOT for production submission.
"""

import asyncio
import argparse
import sys
import os
import json
import csv
import random
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import development utilities
try:
    from dev_utils import MockPipeline, generate_sample_transcripts, generate_mock_evaluation_results, save_results_to_csv, is_demo_mode
    DEV_UTILS_AVAILABLE = True
except ImportError:
    DEV_UTILS_AVAILABLE = False
    print("⚠️  Development utilities not available")

# Try to import real pipelines
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
    print("⚠️  Real pipelines not available")


async def run_pipeline(pipeline_name: str, transcript: str, api_key: str, verbose: bool = False, demo_mode: bool = False) -> None:
    """
    Run a specific pipeline with the given transcript.
    
    Args:
        pipeline_name: Name of the pipeline to run
        transcript: Input transcript text
        api_key: OpenAI API key (not used in demo mode)
        verbose: Enable verbose logging
        demo_mode: Run in demo mode without API calls
    """
    print(f"🚀 Running {pipeline_name}...")
    print(f"📝 Input transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    
    try:
        if demo_mode and DEV_UTILS_AVAILABLE:
            # Use mock pipeline
            pipeline = MockPipeline(pipeline_name, verbose=verbose)
            success, result, context = await pipeline.process(transcript)
            print(f"✅ Success: {success}")
            print(f"📤 Result: {result}")
            print(f"🔍 Context: {context}")
        elif REAL_PIPELINES_AVAILABLE:
            # Use real pipeline
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
        else:
            print("❌ No pipelines available")
            return
                
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


async def run_batch_evaluation(api_key: str, input_file: str, output_file: str, verbose: bool = False, demo_mode: bool = False) -> None:
    """
    Run batch evaluation on multiple transcripts from a file.
    
    Args:
        api_key: OpenAI API key (not used in demo mode)
        input_file: Path to input file with transcripts
        output_file: Path to output file for results
        verbose: Enable verbose logging
        demo_mode: Run in demo mode without API calls
    """
    print(f"📁 Running batch evaluation...")
    print(f"📥 Input file: {input_file}")
    print(f"📤 Output file: {output_file}")
    
    try:
        if demo_mode and DEV_UTILS_AVAILABLE:
            # Generate sample transcripts and mock results
            print("🎭 Running in demo mode - generating sample data")
            transcripts = generate_sample_transcripts()
            print(f"📊 Generated {len(transcripts)} sample transcripts")
            
            # Generate mock evaluation results
            results = generate_mock_evaluation_results(transcripts)
            
            # Save to CSV
            save_results_to_csv(results, output_file)
            
            # Also save detailed results to JSON
            json_output = output_file.replace('.csv', '_detailed.json')
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"📄 Detailed results saved to: {json_output}")
            
        elif REAL_PIPELINES_AVAILABLE:
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
        else:
            print("❌ No pipelines available")
        
    except Exception as e:
        print(f"❌ Error in batch evaluation: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="STT Post-Processing Evaluation Pipeline Runner (Development Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single pipeline (demo mode)
  python src/main_dev.py --pipeline whisper_prompt --transcript "Hello world" --demo
  
  # Run batch evaluation (demo mode)
  python src/main_dev.py --batch --demo --output results.csv --verbose
  
  # Run with real API (requires API key)
  python src/main_dev.py --pipeline fix_transcript --transcript "Sample text" --api-key YOUR_KEY --verbose
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
        help='Input file path (for batch mode, not required in demo mode)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (for batch mode)'
    )
    
    # Configuration
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (not required in demo mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode without requiring API keys'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pipeline and not args.transcript:
        parser.error("--transcript is required when using --pipeline")
    
    if args.batch and not args.output:
        parser.error("--output is required when using --batch")
    
    # Check if demo mode or API key is provided
    if not args.demo and not args.api_key:
        print("⚠️  No API key provided and demo mode not enabled.")
        print("💡 Use --demo flag to run in demonstration mode without API keys")
        print("💡 Or provide --api-key for real pipeline execution")
        return
    
    # Set demo mode if no API key
    demo_mode = args.demo or not args.api_key
    
    if demo_mode:
        print("🎭 Running in DEMO MODE - no API calls will be made")
        print("📊 Sample data will be generated for demonstration purposes")
    
    # Run the appropriate mode
    if args.batch:
        asyncio.run(run_batch_evaluation(args.api_key or "demo", args.input, args.output, args.verbose, demo_mode))
    else:
        asyncio.run(run_pipeline(args.pipeline, args.transcript, args.api_key or "demo", args.verbose, demo_mode))


if __name__ == "__main__":
    main()
