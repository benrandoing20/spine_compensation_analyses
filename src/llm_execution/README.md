# LLM Execution

This directory contains scripts for running LLM experiments and managing API interactions.

## Files

- **`llm_providers.py`** - LLM provider implementations (OpenAI, NVIDIA APIs)
- **`run_experiment.py`** - Main script to run the spine compensation bias experiment
- **`run_baseline.py`** - Script to run baseline experiments without demographic variations
- **`run_batches.sh`** - Shell script to run experiments in manageable batches
- **`merge_results.py`** - Merge multiple result files from batched runs

## Usage

Run from the project root directory:

```bash
# Test with 10 vignettes
python llm_execution/run_experiment.py --test

# Full run with specific models
python llm_execution/run_experiment.py --models gpt-4o llama-3.3-70b

# Run in batches
./llm_execution/run_batches.sh

# Merge results
python llm_execution/merge_results.py
```

## Output

Results are saved to `analysis/results/` directory.

