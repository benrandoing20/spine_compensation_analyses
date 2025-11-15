# Analysis

This directory contains scripts for analyzing experiment results, comparing to baselines, and generating statistical analyses.

## Files

- **`analyze_results.py`** - Analyze experiment results and compute invasiveness index
- **`compare_to_baseline.py`** - Compare full experiment results to baseline
- **`enhanced_analysis.py`** - Enhanced analysis with odds ratios, counts/percentages, and average marginal effects

## Subdirectories

- **`results/`** - Raw experiment results (CSV/JSON files)
- **`comparison/`** - Output from baseline comparison analysis
- **`enhanced_output/`** - Output from enhanced statistical analysis
- **`medication_analysis/`** - Medication-specific analysis results

## Usage

Run from the project root directory:

```bash
# Basic analysis
python analysis/analyze_results.py analysis/results/merged_results.csv

# Compare to baseline
python analysis/compare_to_baseline.py analysis/results/merged_results.csv --output analysis/comparison/

# Enhanced statistical analysis
python analysis/enhanced_analysis.py analysis/results/merged_results.csv --output analysis/enhanced_output/
```

## Output

Analysis outputs include:
- Statistical test results
- Odds ratios
- Average marginal effects
- Comparison tables (CSV)
- Summary statistics

