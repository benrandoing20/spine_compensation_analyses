#!/usr/bin/env python3
"""
Compare full experiment results to baseline.

Usage:
    python compare_to_baseline.py results/merged_results.csv
    python compare_to_baseline.py results/merged_results.csv --output comparison/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Setup plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_separate(csv_path):
    """Load merged results and separate baseline from full experiment."""
    df = pd.read_csv(csv_path)
    
    # Separate baseline and full results
    baseline = df[df['model'].str.startswith('baseline-')].copy()
    full = df[~df['model'].str.startswith('baseline-')].copy()
    
    # Add clean model name to baseline (without 'baseline-' prefix)
    baseline['base_model'] = baseline['model'].str.replace('baseline-', '')
    
    print(f"Loaded {len(df)} total results:")
    print(f"  - Baseline: {len(baseline)} ({len(baseline['model'].unique())} models)")
    print(f"  - Full experiment: {len(full)} ({len(full['model'].unique())} models)")
    
    return baseline, full


def calculate_rates(df, group_cols, outcome_col):
    """Calculate outcome rates by groups."""
    if not group_cols:
        # Overall rate
        return df[outcome_col].value_counts(normalize=True)
    
    # Group rates
    return df.groupby(group_cols)[outcome_col].value_counts(normalize=True).unstack(fill_value=0)


def compare_to_baseline(full, baseline, outcome_col, demographic_col, model_name):
    """Compare demographic-specific rates to baseline for a given model."""
    
    # Filter to specific model
    full_model = full[full['model'] == model_name]
    baseline_model = baseline[baseline['base_model'] == model_name]
    
    if len(baseline_model) == 0:
        print(f"⚠️  No baseline data for model: {model_name}")
        return None
    
    # Get baseline counts for chi-squared test
    baseline_counts = baseline_model[outcome_col].value_counts()
    
    # Calculate deviations
    results = []
    for demo_value in full_model[demographic_col].unique():
        demo_data = full_model[full_model[demographic_col] == demo_value][outcome_col]
        demo_counts = demo_data.value_counts()
        
        # Chi-square test comparing FULL distribution (all outcome categories)
        # This tests: "Is the distribution of outcomes for this demographic value
        # significantly different from baseline?"
        # Example: "White" vs baseline compares [Surgery: Yes, Surgery: No] distribution all at once
        all_outcomes = sorted(set(demo_counts.index) | set(baseline_counts.index))
        contingency = np.array([
            [demo_counts.get(o, 0) for o in all_outcomes],
            [baseline_counts.get(o, 0) for o in all_outcomes]
        ])
        
        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        except:
            chi2, p_value, dof = np.nan, np.nan, np.nan
        
        # Now store individual outcome rates for reference
        for outcome_value in demo_data.unique():
            demo_pct = (demo_data == outcome_value).mean()
            baseline_pct = (baseline_model[outcome_col] == outcome_value).mean()
            
            abs_dev = demo_pct - baseline_pct
            # Calculate relative deviation as percentage change from baseline
            if baseline_pct > 0:
                rel_dev = (abs_dev / baseline_pct * 100)
            else:
                rel_dev = np.nan
            
            results.append({
                'model': model_name,
                'demographic': demographic_col,
                'demographic_value': demo_value,
                'outcome': outcome_col,
                'outcome_value': outcome_value,
                'baseline_rate': baseline_pct,
                'demographic_rate': demo_pct,
                'absolute_deviation': abs_dev,
                'relative_deviation_pct': rel_dev,
                'chi2': chi2,  # Same for all outcome values within this demographic value
                'p_value': p_value,  # Same for all outcome values within this demographic value
                'dof': dof,  # Degrees of freedom
                'n_demographic': len(demo_data),  # Sample size for demographic
                'n_baseline': len(baseline_model),  # Sample size for baseline
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            })
    
    return pd.DataFrame(results)


def get_expected_outcome_values(outcome):
    """Return the complete list of expected outcome values from the prompt.
    
    This ensures tables include ALL possible values (even if they have 0 counts),
    providing a complete view of what the models chose not to select.
    """
    expected_values = {
        'Medication prescription': [
            'OTC only',
            'Prescription non-opioid',
            'Oral Steroid',
            'Steroid injection',
            'Opioid'
        ],
        'work_status': [
            'Full duty',
            'Modified duty',
            'Off work/Temporary Total Disability'
        ],
        'mental_health_referral': [
            'No referral',
            'Optional counseling',
            'Formal psych/mental health evaluation'
        ],
        'physical_therapy': [
            'No PT ordered',
            'PT ordered'
        ],
        'surgical_referral': [
            'No',
            'Yes'
        ]
    }
    
    # For TTD duration, use what's in the data (numeric)
    if 'duration' in outcome.lower() or 'ttd' in outcome.lower():
        return None  # Will use data values
    
    return expected_values.get(outcome, None)


def create_demographic_summary_tables(comparison_df, baseline_df, full_df, output_dir):
    """Create summary tables with DEMOGRAPHIC as primary grouping, models as rows.
    
    Structure:
    AGE BAND:
      Model    | Baseline          | young             | old
      gpt-4o   | count (percent)   | count (percent)   | count (percent)
      llama    | count (percent)   | count (percent)   | count (percent)
    """
    outcomes = comparison_df['outcome'].unique()
    demographics = sorted(comparison_df['demographic'].unique())
    models = sorted(comparison_df['model'].unique())
    
    for outcome in outcomes:
        print(f"\n📋 Creating summary table for: {outcome}")
        
        # Build table data - organized by demographic, then models as rows
        table_data = []
        
        for demographic in demographics:
            # Get all demographic values for this factor
            demo_values = sorted(comparison_df[comparison_df['demographic'] == demographic]['demographic_value'].unique())
            
            # Create one row per model for this demographic
            for model in models:
                # Get baseline data for this model
                baseline_model = baseline_df[baseline_df['base_model'] == model]
                full_model = full_df[full_df['model'] == model]
                
                if len(baseline_model) == 0:
                    continue
                
                # Get all possible outcome values - use expected values if defined, otherwise use data
                expected_values = get_expected_outcome_values(outcome)
                if expected_values is not None:
                    all_outcome_values = expected_values
                else:
                    all_outcome_values = sorted(full_df[outcome].unique())
                
                # Check if this is a TTD duration outcome (numeric weeks)
                is_ttd_duration = 'duration' in outcome.lower() or 'ttd' in outcome.lower()
                
                # Baseline statistics
                if is_ttd_duration:
                    # For TTD duration, calculate average weeks
                    baseline_values = pd.to_numeric(baseline_model[outcome], errors='coerce')
                    baseline_avg = baseline_values.mean()
                    baseline_total = baseline_values.notna().sum()
                    baseline_dist = f"{baseline_avg:.2f}"
                else:
                    baseline_outcome_counts = baseline_model[outcome].value_counts()
                    baseline_total = len(baseline_model)
                    # Format: (count, percent%) for each outcome value
                    baseline_dist = " / ".join([f"({baseline_outcome_counts.get(ov, 0)}, {baseline_outcome_counts.get(ov, 0)/baseline_total*100:.1f}%)" 
                                                for ov in all_outcome_values])
                
                row_data = {
                    'Demographic': demographic,
                    'Model': model,
                    'Baseline': baseline_dist
                }
                
                # Add each demographic value as a column
                demo_data = comparison_df[
                    (comparison_df['model'] == model) & 
                    (comparison_df['outcome'] == outcome) & 
                    (comparison_df['demographic'] == demographic)
                ]
                
                for demo_value in demo_values:
                    demo_value_data = demo_data[demo_data['demographic_value'] == demo_value]
                    
                    if len(demo_value_data) == 0:
                        continue
                    
                    # Get p-value
                    p_value = demo_value_data.iloc[0]['p_value']
                    
                    # Get counts for this demographic value
                    demo_subset = full_model[full_model[demographic] == demo_value]
                    
                    # Format: different for TTD duration vs categorical outcomes
                    if is_ttd_duration:
                        # For TTD duration, calculate average weeks
                        demo_values_numeric = pd.to_numeric(demo_subset[outcome], errors='coerce')
                        demo_avg = demo_values_numeric.mean()
                        demo_n = demo_values_numeric.notna().sum()
                        demo_dist = f"{demo_avg:.2f}"
                    else:
                        demo_outcome_counts = demo_subset[outcome].value_counts()
                        demo_total = len(demo_subset)
                        # Format: (count, percent%) for each outcome value
                        demo_dist = " / ".join([f"({demo_outcome_counts.get(ov, 0)}, {demo_outcome_counts.get(ov, 0)/demo_total*100:.1f}%)" 
                                                for ov in all_outcome_values])
                    
                    # Significance marker
                    sig_marker = ""
                    if not np.isnan(p_value):
                        if p_value < 0.001:
                            sig_marker = " ***"
                        elif p_value < 0.01:
                            sig_marker = " **"
                        elif p_value < 0.05:
                            sig_marker = " *"
                    
                    # Capitalize demographic value for column header
                    demo_value_cap = demo_value.replace('_', ' ').title()
                    row_data[demo_value_cap] = demo_dist + sig_marker
                
                table_data.append(row_data)
        
        # Convert to DataFrame
        if table_data:
            table_df = pd.DataFrame(table_data)
            
            # Sort by Demographic, then Model
            table_df = table_df.sort_values(['Demographic', 'Model'])
            
            # Save as CSV - restructured to avoid sparse columns
            # Create separate sections for each demographic with only relevant columns
            filename = f"summary_table_{outcome.replace(' ', '_').replace('/', '_').replace(',', '')}.csv"
            output_path = output_dir / filename
            
            with open(output_path, 'w') as f:
                # Check if this is a TTD duration outcome
                is_ttd_duration = 'duration' in outcome.lower() or 'ttd' in outcome.lower()
                
                # Write outcome legend at the top - use expected values to show ALL options
                expected_vals = get_expected_outcome_values(outcome)
                if expected_vals is not None:
                    all_outcome_values = expected_vals
                else:
                    all_outcome_values = sorted(full_df[outcome].unique())
                
                f.write(f"OUTCOME: {outcome}\n")
                if is_ttd_duration:
                    f.write(f"Cell format: X.XX (average weeks)\n")
                    f.write(f"Each cell shows the mean TTD duration in weeks for that group\n")
                else:
                    f.write(f"Cell format: " + " / ".join([f"[{i+1}] {ov}" for i, ov in enumerate(all_outcome_values)]) + "\n")
                    f.write(f"Each cell shows: (count, percent%) for each outcome in order above\n")
                    f.write(f"Example: (100, 50.0%) / (100, 50.0%) means 100 (50%) for outcome [1], 100 (50%) for outcome [2]\n")
                    f.write(f"NOTE: Categories with (0, 0.0%) were never selected by any model.\n")
                f.write("\n")
                
                for demographic in demographics:
                    demo_data = table_df[table_df['Demographic'] == demographic].copy()
                    
                    if len(demo_data) == 0:
                        continue
                    
                    # Get only the columns with data for this demographic (non-null)
                    cols_with_data = ['Model', 'Baseline']
                    for col in demo_data.columns:
                        if col not in ['Demographic', 'Model', 'Baseline'] and demo_data[col].notna().any():
                            cols_with_data.append(col)
                    
                    # Add demographic header row
                    f.write(f"\n{demographic.upper().replace('_', ' ')}\n")
                    
                    # Write just the relevant columns for this demographic
                    demo_subset = demo_data[cols_with_data]
                    demo_subset.to_csv(f, index=False)
                    
            print(f"   ✅ Saved: {filename}")
            
            # Also save a more readable version with better formatting
            readable_filename = f"summary_table_{outcome.replace(' ', '_').replace('/', '_').replace(',', '')}_readable.txt"
            readable_path = output_dir / readable_filename
            with open(readable_path, 'w') as f:
                # Check if this is a TTD duration outcome
                is_ttd_duration = 'duration' in outcome.lower() or 'ttd' in outcome.lower()
                
                # Get all outcome values for legend - use expected values to show ALL options
                expected_vals = get_expected_outcome_values(outcome)
                if expected_vals is not None:
                    all_outcome_values = expected_vals
                else:
                    all_outcome_values = sorted(full_df[outcome].unique())
                
                f.write(f"SUMMARY TABLE: {outcome}\n")
                f.write("=" * 180 + "\n")
                f.write(f"Baseline N = 100 for all models (100 replicates)\n")
                
                if is_ttd_duration:
                    f.write("\nCELL FORMAT:\n")
                    f.write("  Each cell shows: X.XX (average weeks)\n")
                    f.write("  Where X.XX is the mean TTD duration in weeks for that group\n")
                else:
                    f.write("\nCELL FORMAT LEGEND:\n")
                    for i, ov in enumerate(all_outcome_values):
                        f.write(f"  [{i+1}] {ov}\n")
                    f.write("\nEach cell shows: (count, percent%) for outcome [1] / (count, percent%) for outcome [2] / ...\n")
                    f.write("Example: (100, 50.0%) / (100, 50.0%) means 100 (50%) for outcome [1], 100 (50%) for outcome [2]\n")
                    f.write("\nNOTE: Categories with (0, 0.0%) were never selected by any model.\n")
                
                f.write("Significance: *** p<0.001, ** p<0.01, * p<0.05\n")
                f.write("=" * 180 + "\n\n")
                
                # Write grouped by DEMOGRAPHIC first, models as rows
                for demographic in demographics:
                    demo_data = table_df[table_df['Demographic'] == demographic]
                    
                    if len(demo_data) == 0:
                        continue
                    
                    f.write(f"\n{'='*180}\n")
                    f.write(f"{demographic.upper().replace('_', ' ')}\n")
                    f.write(f"{'='*180}\n\n")
                    
                    # Column headers - Model, Baseline, then demographic values (capitalized)
                    f.write(f"{'Model':<20} | {'Baseline':<50}")
                    for col in demo_data.columns:
                        if col not in ['Demographic', 'Model', 'Baseline'] and demo_data[col].notna().any():
                            col_display = str(col).replace('_', ' ').title() if isinstance(col, str) else str(col)
                            f.write(f" | {col_display:<50}")
                    f.write("\n")
                    f.write("-" * 180 + "\n")
                    
                    # Data rows - one per model
                    for _, row in demo_data.iterrows():
                        f.write(f"{row['Model']:<20} | {row['Baseline']:<50}")
                        for col in demo_data.columns:
                            if col not in ['Demographic', 'Model', 'Baseline'] and pd.notna(row[col]):
                                f.write(f" | {row[col]:<50}")
                        f.write("\n")
                    
                    f.write("\n")
            
            print(f"   ✅ Saved readable version: {readable_filename}")


def plot_comparison(comparison_df, outcome_col, demographic_col, output_dir):
    """Create grouped bar plot showing all outcome categories for each demographic group and model."""
    
    # Get unique demographic values, models, and outcome values
    demographic_values = sorted(comparison_df['demographic_value'].unique())
    models = sorted(comparison_df['model'].unique())
    
    # Filter outcome_values to only include those with meaningful data
    # This prevents empty/near-empty subplots with excessive white space
    outcome_values = []
    for outcome_val in sorted(comparison_df['outcome_value'].unique()):
        outcome_df = comparison_df[comparison_df['outcome_value'] == outcome_val]
        # Include if there's data and at least some non-zero deviations
        if len(outcome_df) > 0:
            max_abs_deviation = outcome_df['absolute_deviation'].abs().max()
            # Only include if max absolute deviation is > 0.01 (1%)
            if max_abs_deviation > 0.01:
                outcome_values.append(outcome_val)
    
    if len(comparison_df) == 0 or len(outcome_values) == 0:
        return
    
    # Determine figure size based on number of categories
    n_demos = len(demographic_values)
    n_outcomes = len(outcome_values)
    n_models = len(models)
    fig_width = max(16, n_demos * 2.5)
    # Adjust height based on number of outcome categories
    if n_outcomes == 2:
        fig_height = 8
    elif n_outcomes == 3:
        fig_height = 10
    else:
        fig_height = max(8, min(n_outcomes * 2, 16))
    
    # Create figure with subplots (one per outcome category)
    fig, axes = plt.subplots(n_outcomes, 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    
    # Distinct color palette for models - using qualitative colors
    model_colors = {
        'gpt-4o': '#1f77b4',           # blue
        'llama-3.3-70b': '#ff7f0e',    # orange
        'mistral-medium-3': '#2ca02c',  # green
        'qwen3-next-80b': '#d62728'    # red
    }
    
    # Plot each outcome category in its own subplot
    for outcome_idx, outcome_value in enumerate(outcome_values):
        ax = axes[outcome_idx]
        plot_df = comparison_df[comparison_df['outcome_value'] == outcome_value].copy()
        
        if len(plot_df) == 0:
            ax.set_visible(False)
            continue
        
        # Set up grouped bars
        x = np.arange(len(demographic_values))
        n_models = len(models)
        width = 0.75 / n_models  # Slightly narrower for better separation
        
        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = plot_df[plot_df['model'] == model]
            
            # Get deviations in order of demographic_values
            deviations = []
            significances = []
            for demo_val in demographic_values:
                demo_row = model_data[model_data['demographic_value'] == demo_val]
                if len(demo_row) > 0:
                    deviations.append(demo_row.iloc[0]['absolute_deviation'] * 100)
                    significances.append(demo_row.iloc[0]['significant'])
                else:
                    deviations.append(0)
                    significances.append(False)
            
            # Plot bars with proper positioning - ensure clear separation
            offset = (i - n_models/2 + 0.5) * width
            positions = x + offset
            
            bars = ax.bar(positions, deviations, width * 0.9,  # 90% of width for clear gaps
                         label=model if outcome_idx == 0 else "", 
                         color=model_colors.get(model, '#7f7f7f'), 
                         alpha=0.9, 
                         edgecolor='black', 
                         linewidth=1.2,
                         zorder=3)  # Ensure bars are drawn on top
            
            # Add value labels and significance stars
            for j, (pos, dev, sig) in enumerate(zip(positions, deviations, significances)):
                # Always add value label (show 0.00 for zero deviations)
                if abs(dev) > 0.01:
                    label_offset = 1.5 if dev > 0 else -1.5
                    label_va = 'bottom' if dev > 0 else 'top'
                else:
                    # For zero values, place label just above the zero line
                    label_offset = 0.8
                    label_va = 'bottom'
                
                ax.text(pos, dev + label_offset, f'{dev:.2f}', 
                       ha='center', va=label_va,
                       fontsize=7, fontweight='normal', color='black', zorder=4)
                
                # Add significance star
                if sig:
                    star_offset = 4 if dev > 0 else -4
                    ax.text(pos, dev + star_offset, '★', 
                           ha='center', va='bottom' if dev > 0 else 'top',
                           fontsize=8, fontweight='bold', color='gold', zorder=5)
        
        # Formatting for this subplot
        ax.set_ylabel(f'{outcome_value}\n(pp)', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        
        # Only show x-labels on bottom subplot
        if outcome_idx == n_outcomes - 1:
            ax.set_xticklabels(demographic_values, rotation=45, ha='right', fontsize=10)
            ax.set_xlabel(demographic_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.9)
        ax.grid(axis='y', alpha=0.4, linestyle='--', zorder=0)
        ax.grid(axis='x', alpha=0.2, linestyle=':', zorder=0)
        ax.set_axisbelow(True)  # Grid behind bars
        
        # Legend only on top subplot - positioned outside to not overlap
        if outcome_idx == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), fontsize=10, framealpha=0.95, 
                     ncol=1, title='Model', title_fontsize=11, edgecolor='black', fancybox=True)
    
    # Overall title
    fig.suptitle(f'{outcome_col.replace("_", " ").title()}: All Categories\nDeviation from Baseline by {demographic_col.replace("_", " ").title()}',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Add info box at bottom
    fig.text(0.99, 0.01, "★ = significant (p<0.05) | pp = percentage points", 
             ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0.02, 0.88, 0.98], h_pad=2)  # Leave space on right for legend, add vertical spacing
    
    # Save - sanitize filename by replacing special characters
    safe_outcome = outcome_col.replace(' ', '_').replace('/', '_').replace(',', '')
    safe_demographic = demographic_col.replace(' ', '_').replace('/', '_').replace(',', '')
    output_path = Path(output_dir) / f'{safe_outcome}_{safe_demographic}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results to baseline")
    parser.add_argument('csv_file', help='Path to merged results CSV')
    parser.add_argument('--output', default='analysis/comparison', help='Output directory')
    parser.add_argument('--model', help='Specific model to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON ANALYSIS")
    print("=" * 60 + "\n")
    
    baseline, full = load_and_separate(args.csv_file)
    
    # Define outcomes and demographics to analyze
    outcomes = ['surgical_referral', 'work_status', 'Medication prescription', 
                'mental_health_referral', 'physical_therapy', 
                'If Off work/Temporary Total Disability, duration in weeks']
    
    demographics = ['age_band', 'race_ethnicity', 'gender_identity', 
                   'sexual_orientation', 'socioeconomic_status', 'occupation_type',
                   'language_proficiency', 'geography']
    
    # Get models to analyze
    if args.model:
        models = [args.model]
    else:
        models = full['model'].unique()
    
    # Run comparisons
    all_comparisons = []
    
    for model in models:
        print(f"\n📊 Analyzing model: {model}")
        print("-" * 60)
        
        for outcome in outcomes:
            if outcome not in full.columns:
                continue
                
            for demographic in demographics:
                if demographic not in full.columns:
                    continue
                
                comp_df = compare_to_baseline(full, baseline, outcome, demographic, model)
                
                if comp_df is not None and len(comp_df) > 0:
                    all_comparisons.append(comp_df)
                    
                    # Print summary of significant findings
                    sig_findings = comp_df[comp_df['significant']]
                    if len(sig_findings) > 0:
                        print(f"\n  {outcome} × {demographic}:")
                        for _, row in sig_findings.iterrows():
                            print(f"    • {row['demographic_value']}: "
                                 f"{row['outcome_value']} = {row['demographic_rate']:.1%} "
                                 f"(baseline: {row['baseline_rate']:.1%}, "
                                 f"Δ = {row['absolute_deviation']:+.1%}, "
                                 f"p = {row['p_value']:.4f})")
    
    # Combine all comparisons
    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        
        # Save detailed results (all outcome values listed separately)
        output_csv = output_dir / 'baseline_comparison_detailed.csv'
        combined.to_csv(output_csv, index=False)
        print(f"\n✅ Saved detailed comparison: {output_csv}")
        
        # Save summary of significant findings
        sig_only = combined[combined['significant']]
        summary_csv = output_dir / 'baseline_comparison_significant.csv'
        sig_only.to_csv(summary_csv, index=False)
        print(f"✅ Saved significant findings: {summary_csv}")
        
        # Create demographic summary tables (one per outcome)
        print("\n📊 Creating demographic summary tables...")
        create_demographic_summary_tables(combined, baseline, full, output_dir)
        
        # Generate plots
        print("\n📊 Generating comparison plots...")
        for outcome in outcomes:
            if outcome not in combined['outcome'].values:
                continue
            for demographic in demographics:
                if demographic not in combined['demographic'].values:
                    continue
                
                outcome_demo_df = combined[
                    (combined['outcome'] == outcome) & 
                    (combined['demographic'] == demographic)
                ]
                
                if len(outcome_demo_df) > 0:
                    plot_comparison(outcome_demo_df, outcome, demographic, output_dir)
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"\nTotal comparisons: {len(combined)}")
        print(f"Significant findings: {len(sig_only)} ({len(sig_only)/len(combined)*100:.1f}%)")
        print(f"\nBy outcome:")
        print(sig_only['outcome'].value_counts())
        print(f"\nBy demographic:")
        print(sig_only['demographic'].value_counts())
        
    else:
        print("\n⚠️  No comparison results generated")


if __name__ == "__main__":
    main()

