"""
Development utilities for testing and demonstration.

This module contains mock implementations and development tools
that are NOT part of the production submission code.
"""

import asyncio
import random
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple


class MockPipeline:
    """Mock pipeline for demonstration purposes when API keys are not available."""
    
    def __init__(self, pipeline_name: str, verbose: bool = False):
        self.pipeline_name = pipeline_name
        self.verbose = verbose
        
    async def process(self, transcript: str) -> Tuple[bool, str, str]:
        """Simulate pipeline processing with realistic delays and outputs."""
        if self.verbose:
            print(f"🤖 Mock {self.pipeline_name} processing: {transcript[:50]}...")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate realistic mock outputs based on pipeline type
        if "whisper" in self.pipeline_name.lower():
            # Generate context-aware prompts
            topics = ["basketball game", "technical discussion", "weather report", "cooking show"]
            entities = ["LeBron James", "Stephen Curry", "API endpoints", "authentication"]
            jargon = ["three-pointer", "OAuth 2.0", "JWT tokens", "rain forecast"]
            
            topic = random.choice(topics)
            entity = random.choice(entities)
            jargon_term = random.choice(jargon)
            
            prompt = f"Context: {topic}. Key entities: {entity}. Domain terms: {jargon_term}."
            context = f"topic: {topic}, entities: [{entity}], jargon: [{jargon_term}]"
            
            return True, prompt, context
            
        elif "fix" in self.pipeline_name.lower():
            # Simulate transcript correction
            corrections = [
                "corrected transcript with improved grammar",
                "enhanced version with better punctuation",
                "refined text with proper capitalization"
            ]
            return True, random.choice(corrections), "correction_context"
            
        elif "names" in self.pipeline_name.lower():
            # Simulate named entity extraction
            names = ["LeBron James", "Stephen Curry", "Draymond Green", "Anthony Davis"]
            extracted = random.sample(names, random.randint(1, 3))
            return True, ", ".join(extracted), "ner_context"
            
        elif "topic" in self.pipeline_name.lower():
            # Simulate topic extraction
            topics = ["NBA basketball game", "Technical software discussion", "Weather forecast", "Cooking tutorial"]
            return True, random.choice(topics), "topic_context"
            
        else:
            return True, "mock_output", "mock_context"


def generate_sample_transcripts() -> List[str]:
    """Generate sample transcripts for demonstration."""
    return [
        "The Lakers are playing against the Warriors tonight. LeBron James and Stephen Curry are both in the starting lineup.",
        "We need to implement the new authentication system using OAuth 2.0 and JWT tokens for secure API access.",
        "The weather forecast shows rain tomorrow with temperatures around 15 degrees Celsius and wind speeds of 20 kilometers per hour.",
        "Golden State Warriors defeated the Los Angeles Lakers 120-115 in overtime. Stephen Curry scored 35 points.",
        "Our team is working on machine learning models for natural language processing and speech recognition systems."
    ]


def generate_mock_evaluation_results(transcripts: List[str]) -> List[Dict[str, Any]]:
    """Generate mock evaluation results that match the expected CSV format."""
    video_ids = [f"video_{i:03d}" for i in range(1, 11)]
    pipelines = [
        "FixTranscriptByLLMPipeline",
        "GenerateWhisperPromptPipeline", 
        "GenerateNamesPipeline",
        "GenerateTopicPipeline"
    ]
    
    results = []
    
    for video_id in video_ids:
        # Generate random number of segments for this video
        num_segments = random.randint(20, 80)
        baseline_wer = random.uniform(0.08, 0.15)
        
        for pipeline in pipelines:
            # Generate realistic improvement statistics
            improved = random.randint(5, int(num_segments * 0.6))
            same = random.randint(5, int(num_segments * 0.4))
            not_improved = num_segments - improved - same
            
            # Ensure non-negative values
            not_improved = max(0, not_improved)
            
            # Calculate percentages
            improved_pct = (improved / num_segments) * 100
            same_pct = (same / num_segments) * 100
            not_improved_pct = (not_improved / num_segments) * 100
            
            # Generate realistic WER improvements
            if pipeline == "GenerateNamesPipeline":
                # Names pipeline typically shows good improvement
                wer_after = max(0.01, baseline_wer - random.uniform(0.02, 0.06))
            elif pipeline == "FixTranscriptByLLMPipeline":
                # Fix pipeline shows moderate improvement
                wer_after = max(0.01, baseline_wer - random.uniform(0.01, 0.04))
            else:
                # Other pipelines show variable results
                wer_after = max(0.01, baseline_wer + random.uniform(-0.03, 0.03))
            
            result = {
                "Video_ID": video_id,
                "Pipeline": pipeline,
                "Total_Segments": num_segments,
                "Improved": improved,
                "Same": same,
                "Not_Improved": not_improved,
                "Improved_%": round(improved_pct, 2),
                "Same_%": round(same_pct, 2),
                "Not_Improved_%": round(not_improved_pct, 2),
                "Mean_WER_Before": round(baseline_wer, 4),
                "Mean_WER_After": round(wer_after, 4)
            }
            results.append(result)
    
    # Add overall summary rows
    for pipeline in pipelines:
        pipeline_results = [r for r in results if r["Pipeline"] == pipeline]
        if pipeline_results:
            total_segments = sum(r["Total_Segments"] for r in pipeline_results)
            total_improved = sum(r["Improved"] for r in pipeline_results)
            total_same = sum(r["Same"] for r in pipeline_results)
            total_not_improved = sum(r["Not_Improved"] for r in pipeline_results)
            
            # Calculate weighted average WER
            total_wer_before = sum(r["Mean_WER_Before"] * r["Total_Segments"] for r in pipeline_results)
            total_wer_after = sum(r["Mean_WER_After"] * r["Total_Segments"] for r in pipeline_results)
            
            mean_wer_before = total_wer_before / total_segments if total_segments > 0 else 0
            mean_wer_after = total_wer_after / total_segments if total_segments > 0 else 0
            
            overall_result = {
                "Video_ID": "OVERALL",
                "Pipeline": pipeline,
                "Total_Segments": total_segments,
                "Improved": total_improved,
                "Same": total_same,
                "Not_Improved": total_not_improved,
                "Improved_%": round((total_improved / total_segments) * 100, 2) if total_segments > 0 else 0,
                "Same_%": round((total_same / total_segments) * 100, 2) if total_segments > 0 else 0,
                "Not_Improved_%": round((total_not_improved / total_segments) * 100, 2) if total_segments > 0 else 0,
                "Mean_WER_Before": round(mean_wer_before, 4),
                "Mean_WER_After": round(mean_wer_after, 4)
            }
            results.append(overall_result)
    
    return results


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to CSV file in the expected format."""
    if not results:
        print("❌ No results to save")
        return
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get fieldnames from the first result
    fieldnames = list(results[0].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Results saved to: {output_file}")
    print(f"📊 Total results: {len(results)}")


def is_demo_mode() -> bool:
    """Check if the system should run in demo mode."""
    # Check for demo environment variable
    import os
    return os.getenv("STT_DEMO_MODE", "false").lower() == "true"
