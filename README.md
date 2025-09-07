# STT Post-Processing Evaluation Project

A comprehensive evaluation system for Speech-to-Text (STT) post-processing using Large Language Models (LLMs). This project loads NBA audio transcripts from Hugging Face and runs 4 specialized pipelines to enhance and analyze the content.

## 🏀 Dataset

- **Source**: [ShiriGilboa/my-nba-dataset](https://huggingface.co/datasets/ShiriGilboa/my-nba-dataset)
- **Content**: 421 NBA audio samples with transcriptions
- **Features**: Audio files + text transcriptions

## 🔧 Pipelines

1. **GenerateWhisperPromptPipeline** - Generates context-aware prompts for Whisper STT
2. **FixTranscriptByLLMPipeline** - Post-processes and corrects STT transcripts  
3. **GenerateNamesPipeline** - Extracts named entities from transcripts
4. **GenerateTopicPipeline** - Identifies conversation topics

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Evaluation

```bash
# Run with default settings (10 samples)
python main.py --api-key YOUR_OPENAI_API_KEY

# Run with more samples
python main.py --api-key YOUR_OPENAI_API_KEY --samples 50

# Run with verbose output
python main.py --api-key YOUR_OPENAI_API_KEY --samples 20 --verbose
```

### Using Conda Environment

```bash
# Activate conda environment (if available)
conda activate whisper-llm

# Run the evaluation
python main.py --api-key YOUR_OPENAI_API_KEY
```

## 📊 Output

The script generates:
- **Console output**: Real-time progress and results
- **CSV file**: `data/pipeline_evaluation_results.csv` with detailed results
- **Summary statistics**: Success rates and performance metrics

## 🎛️ Dashboard

Launch the interactive dashboard to visualize results:

```bash
streamlit run src/dashboard/main.py
```

## 📁 Project Structure

```
stt_evaluation_project/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── config.py              # Configuration
├── data/                  # Results and data
└── src/
    ├── core/
    │   └── pipelines.py   # Pipeline implementations
    ├── agents/
    │   └── llm_agents.py  # LLM agent classes
    ├── utils/             # Utility functions
    └── dashboard/
        └── main.py        # Streamlit dashboard
```

## 🔍 Features

- **Dataset Loading**: Automatic loading from Hugging Face
- **Parallel Processing**: Efficient pipeline execution
- **Error Handling**: Robust error management
- **Results Export**: CSV export for analysis
- **Interactive Dashboard**: Visual result exploration
- **Configurable**: Adjustable sample sizes and verbosity

## 📈 Results Analysis

The generated CSV contains:
- Sample metadata (ID, duration, transcript length)
- Pipeline results (success status, output)
- Performance metrics (processing time, result quality)
- Timestamps and source information

## 🛠️ Development

The project is structured for easy extension:
- Add new pipelines by extending `BasePipeline`
- Modify agent instructions in `src/utils/instructions.py`
- Customize evaluation metrics in the main script

## 📝 License

This project is part of an academic evaluation system for STT post-processing research.
