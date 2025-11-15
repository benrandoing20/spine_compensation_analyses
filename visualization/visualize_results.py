#!/usr/bin/env python3
"""
Visualize experiment results with statistical significance plots.

Usage:
    python visualize_results.py results/merged_results.csv
    python visualize_results.py results/merged_results.csv --output plots/
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


def calculate_invasiveness(row):
    """Calculate invasiveness index (0-10 scale)."""
    try:
        score = (
            config.INVASIVENESS_SCORES["medication"].get(row.get("Medication prescription", ""), 0) +
            config.INVASIVENESS_SCORES["work_status"].get(row.get("work_status", ""), 0) +
            config.INVASIVENESS_SCORES["physical_therapy"].get(row.get("physical_therapy", ""), 0) +
            config.INVASIVENESS_SCORES["mental_health_referral"].get(row.get("mental_health_referral", ""), 0) +
            config.INVASIVENESS_SCORES["surgical_referral"].get(row.get("surgical_referral", ""), 0)
        )
        return score
    except:
        return np.nan


def plot_overall_distribution(df, output_dir):
    """Plot overall invasiveness distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(df['invasiveness_index'], bins=11, range=(-0.5, 10.5), 
                    edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Invasiveness Index')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Invasiveness Index')
    axes[0, 0].axvline(df['invasiveness_index'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {df["invasiveness_index"].mean():.2f}')
    axes[0, 0].legend()
    
    # By model
    df.boxplot(column='invasiveness_index', by='model', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Invasiveness Index')
    axes[0, 1].set_title('Invasiveness by Model')
    axes[0, 1].tick_params(axis='x', rotation=45)
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45, ha='right')
    
    # Violin plot by model
    sns.violinplot(data=df, x='model', y='invasiveness_index', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Invasiveness Index')
    axes[1, 0].set_title('Invasiveness Distribution by Model')
    axes[1, 0].tick_params(axis='x', rotation=45)
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45, ha='right')
    
    # Summary statistics table
    summary = df.groupby('model')['invasiveness_index'].agg(['mean', 'std', 'median']).round(2)
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
    plt.savefig(output_dir / '01_overall_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 01_overall_distribution.png")
    plt.close()


def plot_by_attribute(df, attribute, output_dir, filename):
    """Plot invasiveness by sociodemographic attribute with significance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    df.boxplot(column='invasiveness_index', by=attribute, ax=axes[0])
    axes[0].set_xlabel(attribute.replace('_', ' ').title())
    axes[0].set_ylabel('Invasiveness Index')
    axes[0].set_title(f'Invasiveness by {attribute.replace("_", " ").title()}')
    axes[0].tick_params(axis='x', rotation=45)
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Mean with error bars
    summary = df.groupby(attribute)['invasiveness_index'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = 1.96 * summary['se']
    
    x_pos = np.arange(len(summary))
    axes[1].bar(x_pos, summary['mean'], yerr=summary['ci'], 
               capsize=5, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(summary.index, rotation=45, ha='right')
    axes[1].set_xlabel(attribute.replace('_', ' ').title())
    axes[1].set_ylabel('Mean Invasiveness Index')
    axes[1].set_title(f'Mean Invasiveness (95% CI) by {attribute.replace("_", " ").title()}')
    
    # Add significance test result
    groups = [group['invasiveness_index'].dropna().values 
             for name, group in df.groupby(attribute)]
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


def plot_heatmap(df, output_dir):
    """Plot heatmap of invasiveness by race and gender."""
    pivot = df.pivot_table(
        values='invasiveness_index',
        index='race_ethnicity',
        columns='gender_identity',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', 
               center=5, cbar_kws={'label': 'Mean Invasiveness'})
    plt.title('Mean Invasiveness Index by Race and Gender Identity')
    plt.xlabel('Gender Identity')
    plt.ylabel('Race/Ethnicity')
    plt.tight_layout()
    plt.savefig(output_dir / '08_race_gender_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 08_race_gender_heatmap.png")
    plt.close()


def plot_outcome_frequencies(df, output_dir):
    """Plot categorical outcome frequencies."""
    outcomes = {
        'Medication prescription': 'medication',
        'work_status': 'work_status',
        'surgical_referral': 'surgical_referral',
        'mental_health_referral': 'mental_health'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (outcome, short_name) in enumerate(outcomes.items()):
        if outcome in df.columns:
            counts = df[outcome].value_counts()
            axes[idx].barh(range(len(counts)), counts.values)
            axes[idx].set_yticks(range(len(counts)))
            axes[idx].set_yticklabels([str(x)[:30] for x in counts.index])
            axes[idx].set_xlabel('Count')
            axes[idx].set_title(outcome.replace('_', ' ').title())
            
            # Add percentages
            for i, v in enumerate(counts.values):
                pct = 100 * v / counts.sum()
                axes[idx].text(v, i, f' {pct:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / '09_outcome_frequencies.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 09_outcome_frequencies.png")
    plt.close()


def plot_significance_summary(df, output_dir):
    """Create summary plot of statistical significance across attributes."""
    attributes = ['age_band', 'race_ethnicity', 'gender_identity', 
                  'sexual_orientation', 'socioeconomic_status', 
                  'occupation_type', 'language_proficiency', 'geography']
    
    results = []
    for attr in attributes:
        if attr in df.columns:
            groups = [group['invasiveness_index'].dropna().values 
                     for name, group in df.groupby(attr)]
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
    ax.set_title('Statistical Significance of Invasiveness Differences\n(ANOVA across attribute levels)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '10_significance_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 10_significance_summary.png")
    plt.close()


def plot_by_attribute_per_model(df, attribute, output_dir, filename_prefix):
    """Plot invasiveness by attribute, separated by model."""
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    # Create subplots - 2 columns
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        model_df = df[df['model'] == model]
        
        # Calculate mean and CI for each attribute level
        summary = model_df.groupby(attribute)['invasiveness_index'].agg(['mean', 'std', 'count'])
        summary['se'] = summary['std'] / np.sqrt(summary['count'])
        summary['ci'] = 1.96 * summary['se']
        
        x_pos = np.arange(len(summary))
        axes[idx].bar(x_pos, summary['mean'], yerr=summary['ci'], 
                     capsize=5, alpha=0.7, edgecolor='black')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(summary.index, rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylabel('Mean Invasiveness Index')
        axes[idx].set_ylim(0, 10)
        axes[idx].set_title(f'{model}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add significance test for this model
        groups = [group['invasiveness_index'].dropna().values 
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
    
    fig.suptitle(f'Invasiveness by {attribute.replace("_", " ").title()} - Per Model Comparison',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / f'{filename_prefix}_per_model.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename_prefix}_per_model.png")
    plt.close()


def plot_model_comparison_heatmap(df, attribute, output_dir, filename):
    """Create heatmap showing mean invasiveness for each model × attribute level."""
    pivot = df.pivot_table(
        values='invasiveness_index',
        index='model',
        columns=attribute,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', 
               center=5, cbar_kws={'label': 'Mean Invasiveness'})
    plt.title(f'Mean Invasiveness Index: Model × {attribute.replace("_", " ").title()}')
    plt.xlabel(attribute.replace('_', ' ').title())
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()


def plot_significance_by_model(df, output_dir):
    """Show which attributes are significant for each model as grouped bar chart."""
    attributes = ['age_band', 'race_ethnicity', 'gender_identity', 
                  'sexual_orientation', 'socioeconomic_status', 
                  'occupation_type', 'language_proficiency', 'geography']
    
    models = sorted(df['model'].unique())
    results = []
    
    for model in models:
        model_df = df[df['model'] == model]
        for attr in attributes:
            if attr in model_df.columns:
                groups = [group['invasiveness_index'].dropna().values 
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
    width = 0.8 / len(models)  # Dynamic width based on number of models
    
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
    ax.set_title('Statistical Significance by Model and Attribute\n(Higher values = more significant bias)', 
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
    plt.savefig(output_dir / '11_significance_by_model.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: 11_significance_by_model.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument('results_file', type=Path, help='Path to results CSV file')
    parser.add_argument('--output', type=Path, default=Path('visualization/plots'), 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"❌ Error: File not found: {args.results_file}")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"📊 Loading results from {args.results_file}...")
    df = pd.read_csv(args.results_file)
    
    # Filter successful queries
    df = df[df['success'] == True].copy()
    
    # Exclude baseline models (use compare_to_baseline.py for baseline analysis)
    df = df[~df['model'].str.startswith('baseline-')].copy()
    
    print(f"✅ Loaded {len(df)} successful queries")
    print(f"   Models: {df['model'].unique().tolist()}")
    print(f"   Vignettes: {df['vignette_id'].nunique()}")
    
    # Calculate invasiveness
    print("\n📈 Calculating invasiveness index...")
    df['invasiveness_index'] = df.apply(calculate_invasiveness, axis=1)
    
    # Generate plots
    print(f"\n🎨 Generating visualizations in {args.output}/...")
    print("=" * 60)
    
    # Overall distribution
    plot_overall_distribution(df, args.output)
    
    # By each attribute - overall
    attributes = [
        ('age_band', '02_by_age'),
        ('race_ethnicity', '03_by_race'),
        ('gender_identity', '04_by_gender'),
        ('sexual_orientation', '05_by_orientation'),
        ('socioeconomic_status', '06_by_ses'),
        ('occupation_type', '07_by_occupation'),
        ('language_proficiency', '08_by_language'),
        ('geography', '09_by_geography'),
    ]
    
    for attr, prefix in attributes:
        if attr in df.columns:
            # Overall plot
            plot_by_attribute(df, attr, args.output, f'{prefix}.png')
            # Per-model plot
            plot_by_attribute_per_model(df, attr, args.output, prefix)
            # Heatmap for model × attribute
            plot_model_comparison_heatmap(df, attr, args.output, f'{prefix}_heatmap.png')
    
    # Race × Gender heatmap
    plot_heatmap(df, args.output)
    
    # Outcome frequencies
    plot_outcome_frequencies(df, args.output)
    
    # Significance summary - overall
    plot_significance_summary(df, args.output)
    
    # Significance by model
    plot_significance_by_model(df, args.output)
    
    print("=" * 60)
    print(f"\n🎉 All visualizations saved to {args.output}/")
    print(f"\nGenerated {len(list(args.output.glob('*.png')))} plots")
    print(f"\n📊 Plot types:")
    print(f"   - Overall distributions and model comparisons")
    print(f"   - Per-attribute analysis (overall + per model + heatmap)")
    print(f"   - Statistical significance summaries")
    print(f"   - Model-specific bias patterns")


if __name__ == "__main__":
    main()

