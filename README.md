# Sociodemographic Bias in LLM Clinical Decision-Making for Spine Injury Cases

Research code for evaluating demographic biases in Large Language Models (LLMs) when making clinical recommendations for workers' compensation spine injury cases.

## 📁 Repository Structure

```
spine_compensation_analyses/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.sh                           # Installation script
│
├── src/                               # Source code
│   ├── config.py                      # Shared configuration
│   ├── llm_execution/                 # LLM experiments
│   │   ├── llm_providers.py           # API integrations (OpenAI, NVIDIA)
│   │   ├── run_baseline.py            # Run baseline (no demographics)
│   │   ├── run_experiment.py          # Run full demographic experiment
│   │   └── merge_results.py           # Combine all results
│   ├── compare_to_baseline.py         # Generate manuscript figures
│   └── enhanced_analysis.py           # Logistic regression analysis
│
└── analysis/                          # Data and outputs
    ├── results/
    │   └── merged_results.csv         # Combined experiment data
    ├── comparison/                    # Output: manuscript figures
    │   ├── medication_distribution_by_model.png
    │   ├── surgical_referral_by_model.png
    │   └── ttd_duration_by_model.png
    └── enhanced_output/               # Output: statistical results
        ├── lr_combined_*.csv          # Logistic regression results
        ├── lr_coefficients_*.csv      # Model coefficients
        ├── lr_pvalues_*.csv           # P-values
        └── lr_significance_*.csv      # Significance markers
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spine_compensation_analyses.git
cd spine_compensation_analyses

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# OpenAI (for GPT-4o)
OPENAI_API_KEY=sk-your-key-here

# NVIDIA API (for Llama, Qwen models)
NVIDIA_API_KEY=nvapi-your-key-here
```

**Get API keys:**
- OpenAI: https://platform.openai.com/api-keys
- NVIDIA: https://build.nvidia.com/ (free tier available)

### 3. Run Experiments (Optional)

*Note: Pre-computed results are available in `analysis/results/merged_results.csv`. Skip to Step 4 to reproduce figures.*

```bash
# Run baseline (no demographics) - ~1 minute
python src/llm_execution/run_baseline.py

# Run full experiment - ~4 hours per model
python src/llm_execution/run_experiment.py --models gpt-4o

# Merge all results
python src/llm_execution/merge_results.py
```

### 4. Generate Manuscript Figures

```bash
# Activate virtual environment
source venv/bin/activate

# Generate all three main figures
python src/compare_to_baseline.py \
  analysis/results/merged_results.csv \
  --output analysis/comparison
```

**Output:** Three publication-ready figures in `analysis/comparison/`:
1. **`surgical_referral_by_model.png`** - Surgical referral rates by demographics
2. **`ttd_duration_by_model.png`** - Temporary total disability duration
3. **`medication_distribution_by_model.png`** - Medication prescription patterns

### 5. Run Statistical Analysis

```bash
# Generate logistic regression results
python src/enhanced_analysis.py \
  analysis/results/merged_results.csv \
  --output analysis/enhanced_output
```

**Output:** Statistical analysis files in `analysis/enhanced_output/`:
- `lr_combined_*.csv` - Full regression results with CIs
- `lr_coefficients_*.csv` - Model coefficients
- `lr_pvalues_*.csv` - P-values for all predictors
- `lr_significance_*.csv` - Significance markers (*, **, ***)

## 📊 Experimental Design

### Models Tested
- **gpt-4o** (OpenAI GPT-4 Omni)
- **llama-3.3-70b** (Meta Llama 3.3 70B)
- **qwen3-next-80b** (Alibaba Qwen3 Next 80B)

### Demographic Factors (8)
- Age: Young, Old
- Race/Ethnicity: Asian, Black, Hispanic/Latino, White
- Gender Identity: Cisgender Man, Cisgender Woman, Transgender Man, Transgender Woman
- Sexual Orientation: Heterosexual, Homosexual
- Socioeconomic Status: Lower, Middle, Upper Class
- Occupation: Blue Collar, White Collar
- Language Proficiency: English Proficient, English Non-Proficient
- Geography: Rural, Suburban, Urban

**Total:** 2,304 unique demographic combinations × 3 replicates = **6,912 queries per model**

### Clinical Outcomes (5)
1. **Medication Prescription** - 5 categories (OTC only → Opioid)
2. **Surgical Referral** - Binary (Yes/No)
3. **Temporary Total Disability Duration** - Continuous (weeks)
4. **Physical Therapy** - Binary (ordered or not)
5. **Mental Health Referral** - 3 categories (none → formal evaluation)

### Clinical Vignette
Standardized L5-S1 disc herniation case:
- 6 weeks post-injury
- Failed conservative treatment
- MRI-confirmed radicular symptoms
- Workers' compensation context
- No red flags or contraindications

## 📈 Key Outputs

### Figure 1: Surgical Referral by Demographics
`analysis/comparison/surgical_referral_by_model.png`

Three-panel plot showing surgical referral rates across demographics:
- **Top row:** GPT-4o and Llama-3.3-70b
- **Bottom:** Qwen3-Next-80b (centered)
- **Significance:** Individual comparisons (*) and ANOVA (†)

### Figure 2: Temporary Total Disability Duration
`analysis/comparison/ttd_duration_by_model.png`

Three-panel plot showing TTD duration differences:
- Independent y-axes for each model
- Llama-3.3-70b features axis break for large baseline
- All numbers to 2 decimal places

### Figure 3: Medication Distribution
`analysis/comparison/medication_distribution_by_model.png`

Horizontal stacked bar chart showing medication patterns:
- GPT-4o only (other models show 100% in one category)
- Significance markers (***) for comparisons to baseline
- ANOVA significance (†) for demographic groups

### Statistical Tables
`analysis/enhanced_output/lr_combined_*.csv`

Logistic regression results for each outcome:
- Coefficients with 95% confidence intervals
- Odds ratios
- P-values and significance levels
- Standard errors and z-statistics

## 🔧 Script Reference

### Core Workflow Scripts

| Script | Purpose | Runtime | Output |
|--------|---------|---------|--------|
| `src/llm_execution/run_baseline.py` | Run baseline (no demographics) | ~1 min | `analysis/results/baseline-{model}/` |
| `src/llm_execution/run_experiment.py` | Run full experiment | ~4 hrs/model | `analysis/results/{model}/` |
| `src/llm_execution/merge_results.py` | Combine all results | <1 min | `analysis/results/merged_results.csv` |
| `src/compare_to_baseline.py` | Generate manuscript figures | ~2 min | `analysis/comparison/*.png` |
| `src/enhanced_analysis.py` | Statistical analysis | ~5 min | `analysis/enhanced_output/*.csv` |

### Script Arguments

#### compare_to_baseline.py
```bash
python src/compare_to_baseline.py CSV_FILE [--output DIR]

Arguments:
  CSV_FILE           Path to merged_results.csv
  --output DIR       Output directory (default: analysis/comparison)
```

#### enhanced_analysis.py
```bash
python src/enhanced_analysis.py CSV_FILE [--output DIR]

Arguments:
  CSV_FILE           Path to merged_results.csv
  --output DIR       Output directory (default: analysis/enhanced_output)
```

#### run_experiment.py
```bash
python src/llm_execution/run_experiment.py [OPTIONS]

Options:
  --models MODEL [MODEL ...]    Models to test (default: all 3)
  --vignette-range "START-END"  Range of vignettes (e.g., "1-500")
  --replicates N                Replicates per vignette (default: 3)
  --test                        Quick test mode (10 vignettes)
```

## 📝 Reproducing Analyses

### Using Pre-Computed Results
The repository includes pre-computed results (`analysis/results/merged_results.csv`). To regenerate figures:

```bash
source venv/bin/activate
python src/compare_to_baseline.py analysis/results/merged_results.csv
python src/enhanced_analysis.py analysis/results/merged_results.csv
```

### Running Full Experiment
To re-run experiments from scratch (~12 hours for all 3 models):

```bash
# 1. Run baseline for all models
python src/llm_execution/run_baseline.py

# 2. Run each model (in parallel if desired)
python src/llm_execution/run_experiment.py --models gpt-4o
python src/llm_execution/run_experiment.py --models llama-3.3-70b
python src/llm_execution/run_experiment.py --models qwen3-next-80b

# 3. Merge results
python src/llm_execution/merge_results.py

# 4. Generate figures and analysis
python src/compare_to_baseline.py analysis/results/merged_results.csv
python src/enhanced_analysis.py analysis/results/merged_results.csv
```

## 🔍 Statistical Methods

### Significance Testing
- **Individual comparisons:** Chi-squared tests comparing each demographic group to baseline
- **Group-level effects:** One-way ANOVA (surgical referral, TTD) or Chi-squared (medication) across demographic categories
- **Regression analysis:** Logistic regression with demographic predictors

### Significance Levels
- `*` p < 0.05
- `**` p < 0.01
- `***` p < 0.001
- `†` ANOVA/Chi-squared p < 0.05
- `††` ANOVA/Chi-squared p < 0.01
- `†††` ANOVA/Chi-squared p < 0.001

## 📦 Dependencies

Core requirements (see `requirements.txt`):
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Statistical tests
- `statsmodels` - Logistic regression
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `openai` - OpenAI API
- `python-dotenv` - Environment variables

## ⚠️ Notes

### File Sizes
- `merged_results.csv`: ~50MB (20,736 rows × 18 columns)
- Total results directory: ~200MB

### Computational Requirements
- **Memory:** ~8GB RAM recommended for full analysis
- **Storage:** ~500MB for all results and outputs
- **Runtime:** 
  - Full experiment: ~12 hours (all 3 models)
  - Analysis only: ~7 minutes

### API Costs (Approximate)
- GPT-4o: ~$50 for full experiment (6,912 queries)
- Llama-3.3-70b: Free (NVIDIA API)
- Qwen3-Next-80b: Free (NVIDIA API)

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: merged_results.csv"
```bash
# Ensure results exist
ls analysis/results/merged_results.csv

# Or regenerate from raw results
python llm_execution/merge_results.py
```

### API Key Errors
```bash
# Check .env file exists
cat .env

# Verify format (no quotes needed)
OPENAI_API_KEY=sk-...
NVIDIA_API_KEY=nvapi-...
```

### Rate Limiting
```bash
# Add delay between queries
python llm_execution/run_experiment.py --models gpt-4o --delay 1.5
```

## 📚 Citation

If you use this code or data, please cite:

```bibtex
@article{yourname2025bias,
  title={Sociodemographic Bias in Large Language Model Clinical Decision-Making for Spine Injury Cases},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025},
  note={Code available at: https://github.com/yourusername/spine_compensation_analyses}
}
```

## 📄 License

[MIT License / Your chosen license]

## 👥 Contact

For questions or issues:
- GitHub Issues: https://github.com/yourusername/spine_compensation_analyses/issues
- Email: your.email@institution.edu

---

**Last Updated:** November 2025  
**Version:** 1.0.0
