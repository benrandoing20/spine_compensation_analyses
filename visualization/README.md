# Visualization

This directory contains scripts for generating visualizations of experiment results.

## Files

- **`visualize_results.py`** - Generate comprehensive visualizations with statistical significance plots
- **`visualize_by_outcome.py`** - Visualize results by individual outcome variables

## Subdirectories

- **`plots/`** - Output directory for generated visualizations
- **`plots_by_outcome/`** - Output directory for outcome-specific visualizations

## Usage

Run from the project root directory:

```bash
# Generate main visualizations
python visualization/visualize_results.py analysis/results/merged_results.csv

# Generate outcome-specific visualizations
python visualization/visualize_by_outcome.py analysis/results/merged_results.csv --output visualization/plots_by_outcome/
```

## Output

Generated plots include:
- Overall invasiveness distribution
- Heatmaps by demographic attributes
- Model-specific comparisons
- Significance summaries
- Outcome frequency distributions
- Race-gender interaction plots

