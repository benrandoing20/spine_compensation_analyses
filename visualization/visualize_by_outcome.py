#!/usr/bin/env python3
"""
Visualize experiment results by individual outcome variables.

Usage:
    python visualize_by_outcome.py results/merged_results.csv --output plots_by_outcome/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))
import config

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# Define outcome scoring (0-5 scale where applicable)
OUTCOME_SCORES = {
    "Medication prescription": {
        "OTC only": 0,
        "Prescription non-opioid": 1,
        "Oral Steroid": 2,
        "Steroid injection": 3,
        "Opioid": 4
    },
    "work_status": {
        "Full duty": 0,
        "Modified duty": 1,
        "Off work/Temporary Total Disability": 2
    },
    "physical_therapy": {
        "No PT ordered": 0,
        "PT ordered": 1
    },
    "mental_health_referral": {
        "No referral": 0,
        "Optional counseling": 1,
        "Formal psych/mental health evaluation": 2
    },
    "surgical_referral": {
        "No": 0,
        "Yes": 1
    },
    "ttd_duration_weeks": "numeric"  # Special case - already numeric (weeks)
}

OUTCOME_LABELS = {
    "Medication prescription": "Medication Prescription Score",
    "work_status": "Work Status Restriction Score",
    "physical_therapy": "Physical Therapy Score",
    "mental_health_referral": "Mental Health Referral Score",
    "surgical_referral": "Surgical Referral Score",
    "ttd_duration_weeks": "TTD Duration (Weeks Off Work)"
}


def encode_outcome(df, outcome_col):
    """Encode categorical outcome as numeric score."""
    # Special handling for TTD duration - it's already numeric
    if outcome_col == "ttd_duration_weeks":
        col_name = "If Off work/Temporary Total Disability, duration in weeks"
        if col_name not in df.columns:
            return None
        
        # Convert to numeric (weeks)
        return pd.to_numeric(df[col_name], errors='coerce')
    
    # Standard categorical encoding
    if outcome_col not in df.columns:
        return None
    
    scores = OUTCOME_SCORES.get(outcome_col, {})
    return df[outcome_col].map(scores)


def plot_overall_distribution_by_outcome(df, outcome_col, outcome_scores, output_dir, prefix=""):
    """Plot overall distribution for a specific outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(outcome_scores.dropna(), bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel(OUTCOME_LABELS.get(outcome_col, outcome_col))
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of {OUTCOME_LABELS.get(outcome_col, outcome_col)}')
    axes[0, 0].axvline(outcome_scores.mean(), color='red', 
                       linestyle='--', label=f'Mean: {outcome_scores.mean():.2f}')
    axes[0, 0].legend()
    
    # By model - boxplot
    df_plot = df.copy()
    df_plot['score'] = outcome_scores
    df_plot = df_plot.dropna(subset=['score'])
    
    df_plot.boxplot(column='score', by='model', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel(OUTCOME_LABELS.get(outcome_col, outcome_col))
    axes[0, 1].set_title(f'{outcome_col.replace("_", " ").title()} by Model')
    axes[0, 1].tick_params(axis='x', rotation=45)
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45, ha='right')
    
    # Violin plot by model
    sns.violinplot(data=df_plot, x='model', y='score', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel(OUTCOME_LABELS.get(outcome_col, outcome_col))
    axes[1, 0].set_title(f'{outcome_col.replace("_", " ").title()} Distribution by Model')
    axes[1, 0].tick_params(axis='x', rotation=45)
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Summary statistics table
    summary = df_plot.groupby('model')['score'].agg(['mean', 'std', 'median']).round(2)
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=summary.values,
                            rowLabels=summary.index,
                            colLabels=summary.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 1].set_title('Summary Statistics by Model', pad=20)
    
    plt.tight_layout()
    safe_name = outcome_col.replace(' ', '_').replace('/', '_').lower()
    if prefix:
        safe_name = f"{prefix}{safe_name}"
    plt.savefig(output_dir / f'{safe_name}_01_overall_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {safe_name}_01_overall_distribution.png")
    plt.close()


def plot_by_attribute_by_outcome(df, outcome_scores, outcome_col, attribute, output_dir, filename):
    """Plot outcome by sociodemographic attribute."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_plot = df.copy()
    df_plot['score'] = outcome_scores
    df_plot = df_plot.dropna(subset=['score'])
    
    # Box plot
    df_plot.boxplot(column='score', by=attribute, ax=axes[0])
    axes[0].set_xlabel(attribute.replace('_', ' ').title())
    axes[0].set_ylabel(OUTCOME_LABELS.get(outcome_col, outcome_col))
    axes[0].set_title(f'{outcome_col.replace("_", " ").title()} by {attribute.replace("_", " ").title()}')
    axes[0].tick_params(axis='x', rotation=45)
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Mean with error bars
    summary = df_plot.groupby(attribute)['score'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = 1.96 * summary['se']
    
    x_pos = np.arange(len(summary))
    axes[1].bar(x_pos, summary['mean'], yerr=summary['ci'], 
               capsize=5, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(summary.index, rotation=45, ha='right')
    axes[1].set_xlabel(attribute.replace('_', ' ').title())
    axes[1].set_ylabel(f'Mean {OUTCOME_LABELS.get(outcome_col, outcome_col)}')
    axes[1].set_title(f'Mean Score (95% CI) by {attribute.replace("_", " ").title()}')
    
    # Add significance test result
    groups = [group['score'].dropna().values 
             for name, group in df_plot.groupby(attribute)]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        sig_text = f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}'
        if p_value < 0.001:
            sig_text += ' ***'
        elif p_value < 0.01:
            sig_text += ' **'
        elif p_value < 0.05:
            sig_text += ' *'
        axes[1].text(0.5, 0.98, sig_text, transform=axes[1].transAxes,
                    ha='center', va='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat' if p_value < 0.05 else 'lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def plot_by_attribute_per_model_by_outcome(df, outcome_scores, outcome_col, attribute, output_dir, filename_prefix):
    """Plot outcome by attribute, separated by model."""
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    # Create subplots - 2 columns
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    df_plot = df.copy()
    df_plot['score'] = outcome_scores
    df_plot = df_plot.dropna(subset=['score'])
    
    for idx, model in enumerate(models):
        model_df = df_plot[df_plot['model'] == model]
        
        # Calculate mean and CI for each attribute level
        summary = model_df.groupby(attribute)['score'].agg(['mean', 'std', 'count'])
        summary['se'] = summary['std'] / np.sqrt(summary['count'])
        summary['ci'] = 1.96 * summary['se']
        
        x_pos = np.arange(len(summary))
        axes[idx].bar(x_pos, summary['mean'], yerr=summary['ci'], 
                     capsize=5, alpha=0.7, edgecolor='black')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(summary.index, rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylabel(f'Mean {OUTCOME_LABELS.get(outcome_col, outcome_col)}')
        max_score = df_plot['score'].max()
        axes[idx].set_ylim(0, max_score * 1.1)
        axes[idx].set_title(f'{model}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add significance test for this model
        groups = [group['score'].dropna().values 
                 for name, group in model_df.groupby(attribute)]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            sig_text = f'p={p_value:.4f}'
            if p_value < 0.001:
                sig_text += ' ***'
                color = 'darkred'
            elif p_value < 0.01:
                sig_text += ' **'
                color = 'red'
            elif p_value < 0.05:
                sig_text += ' *'
                color = 'orange'
            else:
                color = 'lightgray'
            
            axes[idx].text(0.5, 0.95, sig_text, transform=axes[idx].transAxes,
                         ha='center', va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{outcome_col.replace("_", " ").title()} by {attribute.replace("_", " ").title()} - Per Model Comparison',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_per_model.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename_prefix}_per_model.png")
    plt.close()


def plot_model_comparison_heatmap_by_outcome(df, outcome_scores, outcome_col, attribute, output_dir, filename):
    """Create heatmap showing mean outcome for each model × attribute level."""
    df_plot = df.copy()
    df_plot['score'] = outcome_scores
    df_plot = df_plot.dropna(subset=['score'])
    
    pivot = df_plot.pivot_table(
        values='score',
        index='model',
        columns=attribute,
        aggfunc='mean'
    )
    
    max_score = df_plot['score'].max()
    
    plt.figure(figsize=(12, max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', 
               vmin=0, vmax=max_score, cbar_kws={'label': f'Mean {OUTCOME_LABELS.get(outcome_col, outcome_col)}'})
    plt.title(f'{outcome_col.replace("_", " ").title()}: Model × {attribute.replace("_", " ").title()}')
    plt.xlabel(attribute.replace('_', ' ').title())
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def plot_significance_summary_by_outcome(df, outcome_scores, outcome_col, output_dir, prefix=""):
    """Create summary plot of statistical significance across attributes for this outcome."""
    attributes = ['age_band', 'race_ethnicity', 'gender_identity', 
                  'sexual_orientation', 'socioeconomic_status', 
                  'occupation_type', 'language_proficiency', 'geography']
    
    df_plot = df.copy()
    df_plot['score'] = outcome_scores
    df_plot = df_plot.dropna(subset=['score'])
    
    results = []
    for attr in attributes:
        if attr in df_plot.columns:
            groups = [group['score'].dropna().values 
                     for name, group in df_plot.groupby(attr)]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                results.append({
                    'attribute': attr.replace('_', ' ').title(),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    results_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if sig else 'gray' for sig in results_df['significant']]
    bars = ax.barh(results_df['attribute'], -np.log10(results_df['p_value']), color=colors, alpha=0.7)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
    ax.axvline(-np.log10(0.01), color='darkred', linestyle='--', label='p=0.01 threshold')
    ax.set_xlabel('-log10(p-value)')
    ax.set_ylabel('Sociodemographic Attribute')
    ax.set_title(f'Statistical Significance: {outcome_col.replace("_", " ").title()}\n(ANOVA across attribute levels)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    safe_name = outcome_col.replace(' ', '_').replace('/', '_').lower()
    if prefix:
        safe_name = f"{prefix}{safe_name}"
    plt.savefig(output_dir / f'{safe_name}_significance_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {safe_name}_significance_summary.png")
    plt.close()


def plot_significance_by_model_by_outcome(df, outcome_scores, outcome_col, output_dir, prefix=""):
    """Show which attributes are significant for each model as grouped bar chart."""
    attributes = ['age_band', 'race_ethnicity', 'gender_identity', 
                  'sexual_orientation', 'socioeconomic_status', 
                  'occupation_type', 'language_proficiency', 'geography']
    
    df_plot = df.copy()
    df_plot['score'] = outcome_scores
    df_plot = df_plot.dropna(subset=['score'])
    
    models = sorted(df_plot['model'].unique())
    results = []
    
    for model in models:
        model_df = df_plot[df_plot['model'] == model]
        for attr in attributes:
            if attr in model_df.columns:
                groups = [group['score'].dropna().values 
                         for name, group in model_df.groupby(attr)]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    results.append({
                        'model': model,
                        'attribute': attr.replace('_', ' ').title(),
                        'p_value': p_value,
                        'neg_log_p': -np.log10(max(p_value, 1e-100))
                    })
    
    results_df = pd.DataFrame(results)
    
    # Pivot for grouped bar chart
    pivot = results_df.pivot(index='attribute', columns='model', values='neg_log_p')
    
    # Reorder attributes for better readability
    attr_order = ['Occupation Type', 'Language Proficiency', 'Socioeconomic Status', 
                  'Geography', 'Gender Identity', 'Race Ethnicity', 'Age Band', 'Sexual Orientation']
    attr_order = [a for a in attr_order if a in pivot.index]
    pivot = pivot.reindex(attr_order)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    x = np.arange(len(pivot.index))
    width = 0.8 / len(models)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(pivot.columns):
        offset = (idx - len(models)/2 + 0.5) * width
        values = pivot[model].values
        bars = ax.barh(x + offset, values, width, label=model, 
                      color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
              label='p=0.05 threshold', alpha=0.7)
    ax.axvline(-np.log10(0.01), color='darkred', linestyle='--', linewidth=2, 
              label='p=0.01 threshold', alpha=0.7)
    
    # Labels and formatting
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('-log10(p-value)', fontsize=12)
    ax.set_ylabel('Sociodemographic Attribute', fontsize=12)
    ax.set_title(f'Statistical Significance: {outcome_col.replace("_", " ").title()}\n(By Model and Attribute)', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Add annotation
    ax.text(0.98, 0.02, 'Bars beyond red line (p<0.05) indicate significant bias',
            transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=9, style='italic', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    safe_name = outcome_col.replace(' ', '_').replace('/', '_').lower()
    if prefix:
        safe_name = f"{prefix}{safe_name}"
    plt.savefig(output_dir / f'{safe_name}_significance_by_model.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {safe_name}_significance_by_model.png")
    plt.close()


def analyze_outcome_set(df, outcome_col, attributes, output_dir, prefix=""):
    """Analyze a single outcome with optional prefix for filenames."""
    print(f"\n{'='*60}")
    print(f"🎨 Analyzing: {outcome_col.replace('_', ' ').title()}")
    if prefix:
        print(f"   ({prefix.replace('_', ' ').strip()})")
    print(f"{'='*60}")
    
    # Encode outcome
    outcome_scores = encode_outcome(df, outcome_col)
    if outcome_scores is None or outcome_scores.isna().all():
        print(f"⚠️  Skipping {outcome_col} - no valid data")
        return
    
    # Add prefix to all output filenames
    safe_outcome = outcome_col.replace(' ', '_').replace('/', '_').lower()
    if prefix:
        safe_outcome = f"{prefix}{safe_outcome}"
    
    # Overall distribution
    plot_overall_distribution_by_outcome(df, outcome_col, outcome_scores, output_dir, prefix)
    
    # By each attribute
    for attr, attr_prefix in attributes:
        if attr in df.columns:
            # Overall plot
            plot_by_attribute_by_outcome(df, outcome_scores, outcome_col, attr, 
                                        output_dir, f'{safe_outcome}_{attr_prefix}.png')
            
            # Per-model plot
            plot_by_attribute_per_model_by_outcome(df, outcome_scores, outcome_col, attr, 
                                                   output_dir, f'{safe_outcome}_{attr_prefix}')
            
            # Heatmap
            plot_model_comparison_heatmap_by_outcome(df, outcome_scores, outcome_col, attr, 
                                                    output_dir, f'{safe_outcome}_{attr_prefix}_heatmap.png')
    
    # Significance summary - overall
    plot_significance_summary_by_outcome(df, outcome_scores, outcome_col, output_dir, prefix)
    
    # Significance by model
    plot_significance_by_model_by_outcome(df, outcome_scores, outcome_col, output_dir, prefix)


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results by individual outcomes")
    parser.add_argument('results_file', type=Path, help='Path to results CSV file')
    parser.add_argument('--output', type=Path, default=Path('visualization/plots_by_outcome'), 
                       help='Output directory for plots')
    parser.add_argument('--outcomes', nargs='+', 
                       choices=list(OUTCOME_SCORES.keys()),
                       default=list(OUTCOME_SCORES.keys()),
                       help='Which outcomes to analyze (default: all)')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"❌ Error: File not found: {args.results_file}")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"📊 Loading results from {args.results_file}...")
    df_full = pd.read_csv(args.results_file)
    
    # Filter successful queries
    df_full = df_full[df_full['success'] == True].copy()
    
    # Exclude baseline models (use compare_to_baseline.py for baseline analysis)
    df_full = df_full[~df_full['model'].str.startswith('baseline-')].copy()
    
    print(f"✅ Loaded {len(df_full)} successful queries")
    print(f"   Models: {df_full['model'].unique().tolist()}")
    print(f"   Vignettes: {df_full['vignette_id'].nunique()}")
    
    # Analyze each outcome
    attributes = [
        ('age_band', 'by_age'),
        ('race_ethnicity', 'by_race'),
        ('gender_identity', 'by_gender'),
        ('sexual_orientation', 'by_orientation'),
        ('socioeconomic_status', 'by_ses'),
        ('occupation_type', 'by_occupation'),
        ('language_proficiency', 'by_language'),
        ('geography', 'by_geography'),
    ]
    
    # First, analyze all non-TTD outcomes (only once, with full data)
    non_ttd_outcomes = [o for o in args.outcomes if o != 'ttd_duration_weeks']
    for outcome_col in non_ttd_outcomes:
        analyze_outcome_set(df_full, outcome_col, attributes, args.output, prefix="")
    
    # For TTD duration, analyze both with and without zeros
    if 'ttd_duration_weeks' in args.outcomes:
        print(f"\n{'#'*60}")
        print(f"# TTD DURATION ANALYSIS (WITH ZEROS)")
        print(f"{'#'*60}")
        
        # TTD with zeros included
        analyze_outcome_set(df_full, 'ttd_duration_weeks', attributes, args.output, prefix="ttd_with_zero_")
        
        print(f"\n{'#'*60}")
        print(f"# TTD DURATION ANALYSIS (NONZERO ONLY)")
        print(f"{'#'*60}")
        
        # TTD excluding zeros
        ttd_col = "If Off work/Temporary Total Disability, duration in weeks"
        if ttd_col in df_full.columns:
            df_nonzero = df_full.copy()
            df_nonzero['_ttd_temp'] = pd.to_numeric(df_nonzero[ttd_col], errors='coerce')
            original_count = len(df_nonzero)
            df_nonzero = df_nonzero[df_nonzero['_ttd_temp'] > 0].copy()
            df_nonzero = df_nonzero.drop(columns=['_ttd_temp'])
            print(f"⚠️  Filtered to TTD > 0: {len(df_nonzero)} cases ({100*len(df_nonzero)/original_count:.1f}% of total)")
            
            analyze_outcome_set(df_nonzero, 'ttd_duration_weeks', attributes, args.output, prefix="ttd_nonzero_")
        else:
            print(f"⚠️  Warning: TTD column not found, cannot filter")
    
    print(f"\n{'='*60}")
    print(f"🎉 All visualizations saved to {args.output}/")
    print(f"\nGenerated {len(list(args.output.glob('*.png')))} plots")
    print(f"\n📊 Analysis completed for {len(args.outcomes)} outcomes:")
    for outcome in args.outcomes:
        display_name = outcome.replace('_', ' ').title()
        if outcome == 'ttd_duration_weeks':
            print(f"   - {display_name} (with zeros and nonzero only)")
        else:
            print(f"   - {display_name}")


if __name__ == "__main__":
    main()

