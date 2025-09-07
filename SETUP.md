# Setup Guide

## Environment Setup

### 1. Set up Conda Environment

```bash
conda create -n whisper-llm python=3.11
conda activate whisper-llm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API Key

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Option B: Command Line Argument**
```bash
python main.py --api-key 'your-api-key-here'
```

### 4. Run the Project

```bash
# Basic usage (10 samples)
python main.py

# With more samples
python main.py --samples 30

# With verbose output
python main.py --samples 20 --verbose
```

## Troubleshooting

### API Key Issues
- Make sure your OpenAI API key is valid and has sufficient credits
- Check that the environment variable is set correctly: `echo $OPENAI_API_KEY`

### Dependencies Issues
- Make sure you're in the correct conda environment: `conda activate whisper-llm`
- Reinstall requirements if needed: `pip install -r requirements.txt --force-reinstall`

### Dataset Loading Issues
- Ensure you have internet connection for Hugging Face dataset download
- Check that the dataset name is correct: `ShiriGilboa/my-nba-dataset`
