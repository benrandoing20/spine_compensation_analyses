# Spine Compensation LLM Bias Benchmark

A comprehensive benchmark for evaluating sociodemographic biases in Large Language Models for clinical decision-making in workers' compensation spine injury cases.

## 📋 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Benchmark Design](#benchmark-design)
- [Usage Guide](#usage-guide)
- [Scripts Reference](#scripts-reference)
- [Output Directories](#output-directories)
- [Workflow Examples](#workflow-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

This benchmark evaluates whether LLMs exhibit biased clinical recommendations based on patient demographics using a standardized spine injury vignette. It tests 4 models across 2,304 unique demographic combinations with 3 replicates each (27,648 total queries).

### What Gets Tested

**8 Demographic Factors:**
- Age: young, old
- Race/Ethnicity: Black, White, Hispanic/Latino, Asian
- Gender Identity: cisgender man/woman, transgender man/woman
- Sexual Orientation: heterosexual, homosexual
- Socioeconomic Status: lower, middle, upper class
- Occupation Type: white collar, blue collar
- Language Proficiency: English proficient/non-proficient
- Geography: urban, suburban, rural

**5 Clinical Outcomes:**
- Medication prescription (OTC → Opioid scale)
- Work status (Full duty → Off work/TTD)
- Physical therapy (ordered or not)
- Mental health referral (none → formal evaluation)
- Surgical referral (yes/no)

### Key Features

✅ **Baseline comparison** - Tests with/without demographics to isolate bias  
✅ **Statistical analysis** - Odds ratios, marginal effects, chi-square tests  
✅ **Publication-ready tables** - 5 formatted tables for manuscripts  
✅ **Comprehensive visualizations** - 40+ plots by outcome and demographic  
✅ **Parallel processing** - Run multiple models simultaneously  
✅ **Auto-save & resume** - Checkpoint every 50 queries  

---

## Getting Started

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spine_compensation_analyses.git
cd spine_compensation_analyses

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
./setup.sh
```

### 2. Configure API Keys

Create a `.env` file with your API keys:

```bash
# OpenAI (for GPT-4o)
OPENAI_API_KEY=sk-your-key-here

# NVIDIA API (for open-weight models)
NVIDIA_API_KEY=nvapi-your-key-here

# Optional: Experiment settings
TEMPERATURE=0.2
TOP_P=0.9
NUM_REPLICATES=3
```

**Where to get API keys:**
- OpenAI: https://platform.openai.com/api-keys
- NVIDIA: https://build.nvidia.com/ (free credits available)

### 3. Quick Test

Run a quick test with 10 vignettes to verify setup:

```bash
python llm_execution/run_experiment.py --test
```

This takes ~2 minutes and creates test results in `analysis/results/`.

---

## Benchmark Design

### Full Factorial Design

**Calculation:** 2 × 4 × 4 × 2 × 3 × 2 × 2 × 3 = **2,304 unique vignettes**

Each vignette tests one unique combination of all 8 demographic factors.

### Replication Strategy

- **3 replicates per vignette** (default, configurable)
- Different random seeds for each replicate
- Enables reliability analysis

### Total Queries

- **Per model:** 2,304 vignettes × 3 replicates = 6,912 queries
- **All 4 models:** 27,648 queries
- **With baseline:** 27,660 queries (adds 3 per model)

### Clinical Vignette

Standardized L5-S1 disc herniation case at 6 weeks post-injury:
- Failed conservative treatment
- Radicular symptoms
- MRI-confirmed pathology
- No red flags
- Workers' compensation context

---

## Usage Guide

### Step 1: Run Baseline Experiment

The baseline establishes an unbiased reference point (no demographics):

```bash
# Run baseline for all models (~1 minute)
python run_baseline.py

# Test with one model first
python run_baseline.py --test --models gpt-4o
```

**Output:** `results/baseline-{model}/` with 3 queries per model

### Step 2: Run Full Experiment

Run the full demographic experiment:

```bash
# Quick test (10 vignettes)
python run_experiment.py --test

# Single model
python run_experiment.py --models gpt-4o

# Multiple models
python run_experiment.py --models gpt-4o llama-3.3-70b

# All default models (full benchmark)
python run_experiment.py
```

**Output:** `results/{model}/` with results saved incrementally

**Estimated time:**
- Test mode: ~2 minutes
- Single model full: ~3-5 hours
- All 4 models sequential: ~15-20 hours
- All 4 models parallel: ~5 hours (see [Parallel Execution](#parallel-execution))

### Step 3: Merge Results

Combine all results into a single file:

```bash
# Automatic merging (searches all subdirectories)
python merge_results.py

# Custom output filename
python merge_results.py --output my_results

# Merge specific directory only
python merge_results.py --results-dir results/gpt-4o
```

**Output:** `results/merged_results.csv` (27,660 rows)

### Step 4: Compare to Baseline

Statistical comparison of demographic-specific vs. baseline recommendations:

```bash
python compare_to_baseline.py results/merged_results.csv
```

**Output:** `comparison/` directory
- `baseline_comparison_detailed.csv` - All comparisons
- `baseline_comparison_significant.csv` - Significant findings only
- 40 PNG files - Visualizations by outcome × demographic

### Step 5: Enhanced Statistical Analysis

Logistic regression with odds ratios and marginal effects:

```bash
python enhanced_analysis.py results/merged_results.csv
```

**Output:** `enhanced_output/` directory
- `odds_ratios_all.csv` - All pairwise group comparisons
- `odds_ratios_significant.csv` - Significant only
- `average_marginal_effects.csv` - AME for each demographic level
- `average_marginal_effects_significant.csv` - Significant only
- `medication_group_comparisons_all.csv` - Medication-specific tests
- `medication_group_comparisons_significant.csv` - Significant only

### Step 6: Generate Publication Tables

Create formatted tables for manuscripts:

```bash
python create_publication_tables.py
```

**Output:** `tables/` directory (5 tables)
- `Table1_Baseline_Comparisons.csv`
- `Table2_Sociodemographic_Disparities.csv`
- `Table3_Odds_Ratios.csv`
- `Table4_Average_Marginal_Effects.csv`
- `Table5_Medication_Comparisons.csv`
- `Summary_Statistics.csv`

### Step 7: Additional Analyses (Optional)

```bash
# Basic descriptive statistics
python analyze_results.py results/merged_results.csv

# Invasiveness index visualizations
python visualize_results.py results/merged_results.csv

# Outcome-specific plots
python visualize_by_outcome.py results/merged_results.csv
```

---

## Scripts Reference

### Core Experiment Scripts

| Script | Purpose | Input | Output | Runtime |
|--------|---------|-------|--------|---------|
| `run_baseline.py` | Run baseline (no demographics) | None | `results/baseline-{model}/` | 1 min |
| `run_experiment.py` | Run full demographic experiment | None | `results/{model}/` | 3-5 hrs/model |
| `merge_results.py` | Combine all results | `results/*/` | `results/merged_results.csv` | <1 min |

### Analysis Scripts

| Script | Purpose | Input | Output | Runtime |
|--------|---------|-------|--------|---------|
| `compare_to_baseline.py` | Statistical tests vs. baseline | `merged_results.csv` | `comparison/` | 2 min |
| `enhanced_analysis.py` | Odds ratios & marginal effects | `merged_results.csv` | `enhanced_output/` | 5 min |
| `create_publication_tables.py` | Generate formatted tables | Multiple CSVs | `tables/` | <1 min |
| `analyze_results.py` | Basic descriptive stats | `merged_results.csv` | `analysis/` | 2 min |

### Visualization Scripts

| Script | Purpose | Input | Output | Runtime |
|--------|---------|-------|--------|---------|
| `visualize_results.py` | Invasiveness index plots | `merged_results.csv` | `plots/` | 2 min |
| `visualize_by_outcome.py` | Outcome-specific plots | `merged_results.csv` | `plots_by_outcome/` | 3 min |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `config.py` | Configuration (models, attributes, scoring) |
| `llm_providers.py` | API integrations (OpenAI, NVIDIA) |
| `run_batches.sh` | Automated batch processing |
| `setup.sh` | One-click installation |

---

## Output Directories

### `results/`
Raw experiment data organized by model.

```
results/
├── gpt-4o/
│   ├── results_final_20250113_143022.csv
│   └── results_checkpoint_20250113_142530.json
├── llama-3.3-70b/
├── mistral-medium-3/
├── qwen3-next-80b/
├── baseline-gpt-4o/
├── baseline-llama-3.3-70b/
├── baseline-mistral-medium-3/
├── baseline-qwen3-next-80b/
└── merged_results.csv  # Combined file (27,660 rows)
```

**Key files:**
- `results_final_*.csv` - Complete results for a model
- `results_checkpoint_*.json` - Auto-saved progress (every 50 queries)
- `merged_results.csv` - All models + baseline combined

### `comparison/`
Baseline comparison analysis (Step 4).

```
comparison/
├── baseline_comparison_detailed.csv         # All 703 comparisons
├── baseline_comparison_significant.csv      # Significant only (10)
├── surgical_referral_age_band.png           # 40 visualization files
├── surgical_referral_race_ethnicity.png     # (5 outcomes × 8 demographics)
├── work_status_gender_identity.png
└── ...
```

**Contents:**
- Chi-square tests for each demographic group vs. baseline
- P-values, effect sizes, absolute/relative deviations
- Grouped bar plots showing deviations by model

### `enhanced_output/`
Advanced statistical analysis (Step 5).

```
enhanced_output/
├── odds_ratios_all.csv                      # Pairwise comparisons
├── odds_ratios_significant.csv              # Significant ORs only
├── average_marginal_effects.csv             # AME for each demographic
├── average_marginal_effects_significant.csv
├── medication_group_comparisons_all.csv
└── medication_group_comparisons_significant.csv
```

**Contents:**
- Logistic regression results
- Odds ratios with 95% CI
- Average marginal effects (AME)
- Medication-specific statistical tests

### `tables/`
Publication-ready tables (Step 6).

```
tables/
├── Table1_Baseline_Comparisons.csv          # 10 significant baseline findings
├── Table2_Sociodemographic_Disparities.csv  # 64 disparities
├── Table3_Odds_Ratios.csv                   # 254 significant ORs
├── Table4_Average_Marginal_Effects.csv      # 241 significant AMEs
├── Table5_Medication_Comparisons.csv        # 10 medication findings
└── Summary_Statistics.csv                   # Overview
```

**Format:** Cleaned, formatted, ready for LaTeX/Word tables

### `analysis/`
Basic descriptive statistics (optional).

```
analysis/
├── analysis_merged_results.csv
└── summary_merged_results.txt
```

### `plots/`
Invasiveness index visualizations (optional).

```
plots/
├── invasiveness_by_age_band.png
├── invasiveness_by_race_ethnicity.png
└── ...
```

### `plots_by_outcome/`
Outcome-specific visualizations (optional).

```
plots_by_outcome/
├── surgical_referral_by_age_band.png
├── work_status_by_race_ethnicity.png
├── ttd_with_zero_by_gender_identity.png
├── ttd_nonzero_by_gender_identity.png      # TTD creates 2 versions
└── ...
```

### `logs/`
Execution logs for debugging.

```
logs/
├── baseline_batch_1_1-500.log
└── batch_1_1-500.log
```

---

## Workflow Examples

### Minimal Workflow (Essential Only)

```bash
# 1. Run baseline + full experiment
python run_baseline.py
python run_experiment.py

# 2. Merge and analyze
python merge_results.py
python compare_to_baseline.py results/merged_results.csv
python enhanced_analysis.py results/merged_results.csv
python create_publication_tables.py

# Done! Check tables/ for results
```

**Output:** 5 publication tables + baseline comparison plots

### Complete Workflow (All Analyses)

```bash
# 1. Experiments
python run_baseline.py
python run_experiment.py

# 2. Merge
python merge_results.py

# 3. All analyses
python compare_to_baseline.py results/merged_results.csv
python enhanced_analysis.py results/merged_results.csv
python analyze_results.py results/merged_results.csv

# 4. All visualizations
python visualize_results.py results/merged_results.csv
python visualize_by_outcome.py results/merged_results.csv

# 5. Generate tables
python create_publication_tables.py
```

**Output:** Tables + 3 directories of plots + detailed statistics

### Parallel Execution

Run multiple models simultaneously to reduce total time:

```bash
# Terminal 1
python run_experiment.py --models gpt-4o --output-dir results/gpt-4o

# Terminal 2
python run_experiment.py --models llama-3.3-70b --output-dir results/llama-3.3-70b

# Terminal 3
python run_experiment.py --models mistral-medium-3 --output-dir results/mistral-medium-3

# Terminal 4
python run_experiment.py --models qwen3-next-80b --output-dir results/qwen3-next-80b

# After all complete, merge and analyze
python merge_results.py
python compare_to_baseline.py results/merged_results.csv
python enhanced_analysis.py results/merged_results.csv
python create_publication_tables.py
```

**Time saved:** ~10-15 hours (runs in ~5 hours instead of ~20)

### Batch Processing

For rate-limited APIs or unstable connections:

```bash
# Option 1: Use provided script
./run_batches.sh

# Option 2: Manual batches
python run_experiment.py --vignette-range "1-500"
python run_experiment.py --vignette-range "501-1000"
python run_experiment.py --vignette-range "1001-1500"
python run_experiment.py --vignette-range "1501-2000"
python run_experiment.py --vignette-range "2001-2304"

# Merge all batches
python merge_results.py
```

### Testing Subset

Test with fewer vignettes before full run:

```bash
# 50 vignettes, 2 models, 2 replicates
python run_experiment.py \
  --vignette-range "1-50" \
  --models gpt-4o llama-3.3-70b \
  --replicates 2 \
  --output-dir results/test_run

# Quick analysis
python merge_results.py --results-dir results/test_run
python analyze_results.py results/test_run/merged_results.csv
```

---

## Troubleshooting

### Installation Issues

**Problem:** Missing dependencies
```bash
pip install -r requirements.txt
# Or: pip install pandas numpy scipy statsmodels matplotlib seaborn tqdm openai python-dotenv
```

**Problem:** Python version
- Requires Python 3.8+
- Check: `python --version`

### API Issues

**Problem:** API key errors
```bash
# Check .env file exists and has correct format
cat .env

# Test API keys
python -c "from llm_providers import query_model; print('OpenAI OK')"
```

**Problem:** Rate limiting
```bash
# Increase delay between queries
python run_experiment.py --delay 1.5

# Or run in smaller batches
python run_experiment.py --vignette-range "1-500" --delay 1.0
```

**Problem:** Timeout errors
- Check internet connection
- Verify API key validity
- Try with `--delay 2.0`

### Experiment Issues

**Problem:** Experiment interrupted
```bash
# Results auto-saved every 50 queries in checkpoint files
# Check results/ for partial files
ls results/*/results_checkpoint_*.json

# Resume from where you left off
# Find last completed vignette, then:
python run_experiment.py --vignette-range "1001-2304"
```

**Problem:** Duplicate data after re-running
```bash
# merge_results.py automatically deduplicates
# based on (model, vignette_id, replicate)
python merge_results.py
```

**Problem:** Out of memory
```bash
# Process models separately
python run_experiment.py --models gpt-4o --output-dir results/gpt-4o
# Then merge later
```

### Analysis Issues

**Problem:** "No module named 'statsmodels'"
```bash
pip install statsmodels
```

**Problem:** Warnings about convergence
- Normal for some demographic combinations with low variance
- Check `enhanced_analysis.log` for details
- Results still valid for converged models

**Problem:** Empty tables
- Ensure `merged_results.csv` exists with data
- Check that baseline experiment was run
- Verify column names match expected format

### Visualization Issues

**Problem:** Plots not showing
```bash
# Check output directory was created
ls comparison/
ls plots/

# Verify matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

**Problem:** Missing plot files
- Some combinations may not have data if specific outcome wasn't observed
- Check `comparison/` directory for available plots
- Normal to have fewer than 40 plots if some outcomes are rare

---

## Command-Line Options Reference

### run_experiment.py

```bash
python run_experiment.py [OPTIONS]

Options:
  --models MODEL [MODEL ...]     Models to test (default: all 4)
  --vignette-range "START-END"   Range of vignettes (default: all 2,304)
  --max-vignettes N             Limit total vignettes (for testing)
  --replicates N                Replicates per vignette (default: 3)
  --delay SECONDS               Delay between queries (default: 0.5)
  --output-dir DIR              Output directory (default: results/)
  --test                        Quick test mode (10 vignettes)
  --list-models                 Show available models

Examples:
  python run_experiment.py --test
  python run_experiment.py --models gpt-4o
  python run_experiment.py --vignette-range "1-500" --delay 1.0
  python run_experiment.py --models gpt-4o --output-dir results/gpt4o_only
```

### run_baseline.py

```bash
python run_baseline.py [OPTIONS]

Options:
  --models MODEL [MODEL ...]    Models to test (default: all 4)
  --replicates N               Replicates per model (default: 3)
  --delay SECONDS              Delay between queries (default: 0.5)
  --test                       Quick test mode (1 model, 1 replicate)
  --list-models                Show available models

Examples:
  python run_baseline.py
  python run_baseline.py --test
  python run_baseline.py --models gpt-4o llama-3.3-70b
```

### merge_results.py

```bash
python merge_results.py [OPTIONS]

Options:
  --results-dir DIR            Directory to search (default: results/)
  --output NAME                Output filename (default: merged_results)
  --no-recursive               Don't search subdirectories

Examples:
  python merge_results.py
  python merge_results.py --output final_results
  python merge_results.py --results-dir results/gpt-4o
```

### compare_to_baseline.py

```bash
python compare_to_baseline.py CSV_FILE [OPTIONS]

Arguments:
  CSV_FILE                     Merged results file

Options:
  --output DIR                 Output directory (default: comparison/)
  --model MODEL               Analyze specific model only

Examples:
  python compare_to_baseline.py results/merged_results.csv
  python compare_to_baseline.py results/merged_results.csv --model gpt-4o
```

### enhanced_analysis.py

```bash
python enhanced_analysis.py CSV_FILE [OPTIONS]

Arguments:
  CSV_FILE                     Merged results file

Options:
  --output DIR                 Output directory (default: enhanced_output/)

Examples:
  python enhanced_analysis.py results/merged_results.csv
  python enhanced_analysis.py results/merged_results.csv --output my_analysis/
```

### create_publication_tables.py

```bash
python create_publication_tables.py

No options - reads from comparison/, enhanced_output/, and medication_analysis/
Outputs to tables/
```

---

## Available Models

### Commercial Models
- **gpt-4o** (OpenAI) - Requires paid API key

### Open-Weight Models (via NVIDIA API)
All models below use NVIDIA API (free credits available):

- **llama-3.3-70b** (Meta) - Flagship open model
- **llama-3.1-405b** (Meta) - Largest Llama
- **deepseek-v3.1** (DeepSeek) - Strong reasoning
- **deepseek-r1** (DeepSeek) - Reasoning specialist
- **qwen3-next-80b** (Alibaba) - Multilingual
- **qwq-32b** (Alibaba) - Quality over quantity
- **kimi-k2** (Moonshot) - Long context specialist
- **mistral-medium-3** (Mistral AI) - Balanced
- **mistral-small-3.1** (Mistral AI) - Efficient

**Default models** (used when running without --models):
- gpt-4o, llama-3.3-70b, mistral-medium-3, qwen3-next-80b

**List all available:**
```bash
python run_experiment.py --list-models
```

---

## Citation

If you use this benchmark for your research, please cite:

```bibtex
@software{spine_compensation_benchmark,
  title = {Spine Compensation LLM Bias Benchmark},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/spine_compensation_analyses}
}
```

---

## License

[Add your license - e.g., MIT, Apache 2.0]

---

## Contact

For questions, issues, or contributions:
- Open a GitHub issue
- Email: [your email]
- Twitter: [@yourusername]

---

## Repository Structure

```
spine_compensation_analyses/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.sh                       # Installation script
├── config.py                      # Shared configuration
├── .env                          # API keys (create from .env.example)
│
├── llm_execution/                 # LLM experiment runners
│   ├── README.md                 # LLM execution guide
│   ├── llm_providers.py          # API integrations
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_baseline.py           # Baseline experiment
│   ├── run_batches.sh            # Batch processing helper
│   └── merge_results.py          # Merge results
│
├── analysis/                      # Statistical analysis
│   ├── README.md                 # Analysis guide
│   ├── analyze_results.py        # Basic descriptive stats
│   ├── compare_to_baseline.py    # Baseline statistical tests
│   ├── enhanced_analysis.py      # Odds ratios & marginal effects
│   ├── results/                  # Experiment data
│   │   ├── {model}/             # Per-model results
│   │   ├── baseline-{model}/    # Baseline results
│   │   └── merged_results.csv   # Combined data
│   ├── comparison/               # Baseline comparison output
│   │   ├── baseline_comparison_detailed.csv
│   │   ├── baseline_comparison_significant.csv
│   │   └── *.png                # 40 visualization files
│   ├── enhanced_output/          # Statistical analysis output
│   │   ├── odds_ratios_all.csv
│   │   ├── odds_ratios_significant.csv
│   │   ├── average_marginal_effects.csv
│   │   └── medication_group_comparisons.csv
│   └── medication_analysis/      # Medication-specific analysis
│
├── visualization/                 # Data visualization
│   ├── README.md                 # Visualization guide
│   ├── visualize_results.py      # Invasiveness plots
│   ├── visualize_by_outcome.py   # Outcome-specific plots
│   ├── plots/                    # Invasiveness visualizations
│   └── plots_by_outcome/         # Outcome-specific plots
│
├── rl/                           # Reinforcement learning
│   ├── README.md                 # RL training guide
│   ├── train_fairness_ppo.py     # PPO fairness training
│   └── run_ppo_training.sh       # Training setup script
│
└── logs/                         # Execution logs
```
