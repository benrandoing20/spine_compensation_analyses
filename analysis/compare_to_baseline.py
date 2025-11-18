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
    
    # Filter out mistral models from both baseline and full
    baseline = baseline[~baseline['model'].str.contains('mistral', case=False, na=False)].copy()
    full = full[~full['model'].str.contains('mistral', case=False, na=False)].copy()
    
    # Add clean model name to baseline (without 'baseline-' prefix)
    baseline['base_model'] = baseline['model'].str.replace('baseline-', '')
    
    print(f"Loaded {len(df)} total results (excluding mistral):")
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


def plot_surgical_referral_by_model(baseline, full, output_dir):
    """
    Plot surgical referral 'Yes' percentage with subplots per model.
    X-axis: demographic factors grouped by demographic type
    Y-axis: percentage of 'Yes' surgical referrals
    """
    outcome_col = 'surgical_referral'
    target_value = 'Yes'
    
    models = sorted(full['model'].unique())
    demographics = ['age_band', 'race_ethnicity', 'gender_identity', 
                   'sexual_orientation', 'socioeconomic_status', 'occupation_type',
                   'language_proficiency', 'geography']
    
    # Create triangle arrangement: 2 plots on top, 1 centered below (same size)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Create axes in triangle arrangement
    axes = []
    if len(models) >= 1:
        axes.append(fig.add_subplot(gs[0, 0:2]))  # Top left (spans 2 columns)
    if len(models) >= 2:
        axes.append(fig.add_subplot(gs[0, 2:4]))  # Top right (spans 2 columns)
    if len(models) >= 3:
        axes.append(fig.add_subplot(gs[1, 1:3]))  # Bottom center (spans 2 columns, centered)
    
    model_colors = {
        'gpt-4o': '#1f77b4',
        'llama-3.3-70b': '#ff7f0e',
        'qwen3-next-80b': '#d62728'
    }
    
    # Find global y-axis range for all models (for shared scale)
    all_percentages = []
    for model in models:
        baseline_model = baseline[baseline['base_model'] == model]
        if len(baseline_model) > 0:
            baseline_pct = (baseline_model[outcome_col] == target_value).mean() * 100
            all_percentages.append(baseline_pct)
        
        full_model = full[full['model'] == model]
        for demo in demographics:
            demo_values = sorted(full_model[demo].dropna().unique())
            for demo_value in demo_values:
                demo_data = full_model[full_model[demo] == demo_value]
                demo_pct = (demo_data[outcome_col] == target_value).mean() * 100
                all_percentages.append(demo_pct)
    
    global_y_max = max(all_percentages) * 1.15 if all_percentages else 100
    
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        
        # Get baseline data
        baseline_model = baseline[baseline['base_model'] == model]
        if len(baseline_model) == 0:
            continue
        
        baseline_pct = (baseline_model[outcome_col] == target_value).mean() * 100
        
        # Collect data for all demographics with grouping
        x_labels = ['Baseline']
        percentages = [baseline_pct]
        colors_list = ['#555555']  # Dark gray for baseline
        
        full_model = full[full['model'] == model]
        
        # Build x_positions with gaps between demographic groups (more pronounced spacing)
        x_positions = [0]  # Baseline at position 0
        current_pos = 2.0  # Start first demographic group with gap after baseline
        
        for demo in demographics:
            demo_values = sorted(full_model[demo].dropna().unique())
            for demo_value in demo_values:
                demo_data = full_model[full_model[demo] == demo_value]
                demo_pct = (demo_data[outcome_col] == target_value).mean() * 100
                
                x_labels.append(f"{demo_value}")
                percentages.append(demo_pct)
                colors_list.append(model_colors.get(model, '#888888'))
                x_positions.append(current_pos)
                current_pos += 1  # Space within group
            
            # Add extra gap between demographic groups (increased for more pronounced grouping)
            current_pos += 1.0
        
        # Plot bars
        x_positions = np.array(x_positions)
        bars = ax.bar(x_positions, percentages, width=0.8, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars (very small font size to prevent overlap)
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{pct:.1f}', ha='center', va='bottom', fontsize=4.5)
        
        # Formatting
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        # Show ylabel on left plots (idx 0 and 2)
        if model_idx in [0, 2]:
            ax.set_ylabel('Surgical Referral "Yes" (%)', fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3, linestyle='--')  # Only horizontal grid lines
        ax.grid(axis='x', visible=False)  # Remove vertical grid lines
        ax.set_ylim(0, global_y_max)  # Shared y-axis scale across all subplots
        ax.set_xlim(-0.5, max(x_positions) + 0.5)  # Adjust x-axis limits
    
    output_path = output_dir / 'surgical_referral_by_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path.name}")
    plt.close()


def plot_ttd_duration_by_model(baseline, full, output_dir):
    """
    Plot average TTD duration - only showing qwen results.
    X-axis: demographic factors grouped by demographic type
    Y-axis: average TTD duration in weeks
    """
    outcome_col = 'If Off work/Temporary Total Disability, duration in weeks'
    
    if outcome_col not in full.columns:
        print(f"⚠️  Column '{outcome_col}' not found, skipping TTD duration plot")
        return
    
    # Only show qwen results
    models = ['qwen3-next-80b']
    
    # Filter data to only qwen
    full = full[full['model'].isin(models)].copy()
    baseline = baseline[baseline['base_model'].isin(models)].copy()
    
    if len(full) == 0:
        print(f"⚠️  No qwen data found, skipping TTD duration plot")
        return
    
    demographics = ['age_band', 'race_ethnicity', 'gender_identity', 
                   'sexual_orientation', 'socioeconomic_status', 'occupation_type',
                   'language_proficiency', 'geography']
    
    # Create single plot for qwen only
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    model_idx = 0
    model = 'qwen3-next-80b'
    
    model_colors = {
        'qwen3-next-80b': '#d62728'
    }
    
    # Get baseline data
    baseline_model = baseline[baseline['base_model'] == model]
    if len(baseline_model) == 0:
        print(f"⚠️  No baseline data for qwen, skipping TTD duration plot")
        return
    
    baseline_values = pd.to_numeric(baseline_model[outcome_col], errors='coerce')
    baseline_avg = baseline_values.mean()
    
    # Collect data for all demographics with grouping
    x_labels = ['Baseline']
    averages = [baseline_avg]
    colors_list = ['#555555']  # Dark gray for baseline
    
    full_model = full[full['model'] == model]
    
    # Build x_positions with gaps between demographic groups (more pronounced spacing)
    x_positions = [0]  # Baseline at position 0
    current_pos = 2.0  # Start first demographic group with gap after baseline
    
    for demo in demographics:
        demo_values = sorted(full_model[demo].dropna().unique())
        for demo_value in demo_values:
            demo_data = full_model[full_model[demo] == demo_value]
            demo_numeric = pd.to_numeric(demo_data[outcome_col], errors='coerce')
            demo_avg = demo_numeric.mean()
            
            x_labels.append(f"{demo_value}")
            averages.append(demo_avg)
            colors_list.append(model_colors.get(model, '#888888'))
            x_positions.append(current_pos)
            current_pos += 1  # Space within group
        
        # Add extra gap between demographic groups (increased for more pronounced grouping)
        current_pos += 1.0
    
    # Plot bars
    x_positions = np.array(x_positions)
    bars = ax.bar(x_positions, averages, width=0.8, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars (very small font size)
    for i, (bar, avg) in enumerate(zip(bars, averages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{avg:.1f}', ha='center', va='bottom', fontsize=4.5)
    
    # Formatting
    ax.set_title(f'{model}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average TTD Duration (weeks)', fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3, linestyle='--')  # Only horizontal grid lines
    ax.grid(axis='x', visible=False)  # Remove vertical grid lines
    ax.set_xlim(-0.5, max(x_positions) + 0.5)  # Adjust x-axis limits
    ax.set_ylim(0, max(averages) * 1.15)
    
    plt.tight_layout()
    
    output_path = output_dir / 'ttd_duration_by_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path.name}")
    plt.close()


def plot_medication_distribution_by_model(baseline, full, output_dir):
    """
    Plot medication distribution with horizontal stacked percentage bars.
    Only shows gpt-4o since other models have 100% single medication type.
    """
    outcome_col = 'Medication prescription'
    
    if outcome_col not in full.columns:
        print(f"⚠️  Column '{outcome_col}' not found, skipping medication plot")
        return
    
    # Only plot gpt-4o since other models show 100% for one medication type
    models = ['gpt-4o']
    
    # Filter to only include gpt-4o in the data
    full = full[full['model'] == 'gpt-4o'].copy()
    baseline = baseline[baseline['base_model'] == 'gpt-4o'].copy()
    
    if len(full) == 0:
        print(f"⚠️  No gpt-4o data found, skipping medication plot")
        return
    
    demographics = ['age_band', 'race_ethnicity', 'gender_identity', 
                   'sexual_orientation', 'socioeconomic_status', 'occupation_type',
                   'language_proficiency', 'geography']
    
    # Medication types in order
    med_types = ['OTC only', 'Prescription non-opioid', 'Oral Steroid', 
                 'Steroid injection', 'Opioid']
    
    # Colors for each medication type
    med_colors = {
        'OTC only': '#90EE90',           # Light green
        'Prescription non-opioid': '#87CEEB',  # Sky blue
        'Oral Steroid': '#FFD700',       # Gold
        'Steroid injection': '#FF8C00',  # Dark orange
        'Opioid': '#DC143C'              # Crimson
    }
    
    # Create single plot for gpt-4o only
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    model = 'gpt-4o'
    
    # Get baseline and full model data
    baseline_model = baseline[baseline['base_model'] == model]
    full_model = full[full['model'] == model]
    
    if len(baseline_model) == 0:
        print(f"⚠️  No baseline data for gpt-4o, skipping medication plot")
        return
    
    # Collect all bars data
    y_labels = ['Baseline']
    bar_data = []  # List of dicts with medication percentages
    
    # Baseline percentages
    baseline_counts = baseline_model[outcome_col].value_counts()
    baseline_total = len(baseline_model)
    baseline_pcts = {med: (baseline_counts.get(med, 0) / baseline_total * 100) 
                    for med in med_types}
    bar_data.append(baseline_pcts)
    
    # Add each demographic category
    for demo in demographics:
        demo_values = sorted(full_model[demo].dropna().unique())
        for demo_value in demo_values:
            demo_data = full_model[full_model[demo] == demo_value]
            demo_counts = demo_data[outcome_col].value_counts()
            demo_total = len(demo_data)
            demo_pcts = {med: (demo_counts.get(med, 0) / demo_total * 100) 
                       for med in med_types}
            
            y_labels.append(f"{demo_value}")
            bar_data.append(demo_pcts)
    
    # Create horizontal stacked bars
    y_positions = np.arange(len(y_labels))
    left_accumulator = np.zeros(len(y_labels))
    
    for med_type in med_types:
        widths = [data[med_type] for data in bar_data]
        ax.barh(y_positions, widths, left=left_accumulator, 
               color=med_colors[med_type], label=med_type,
               edgecolor='white', linewidth=0.5)
        
        # Add percentage labels (only if > 3% to avoid clutter)
        for i, width in enumerate(widths):
            if width > 3:
                ax.text(left_accumulator[i] + width/2, y_positions[i], 
                       f'{width:.0f}%', ha='center', va='center', 
                       fontsize=7, fontweight='bold', color='black')
        
        left_accumulator += widths
    
    # Formatting
    ax.set_title(f'{model}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Medication Distribution (%)', fontsize=10)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Baseline at top
    
    # Add demographic group separators
    current_pos = 1  # After baseline
    for demo in demographics:
        demo_values = sorted(full_model[demo].dropna().unique())
        n_values = len(demo_values)
        if n_values > 0:
            # Add horizontal line before this demographic group
            ax.axhline(y=current_pos - 0.5, color='red', linestyle='--', 
                      alpha=0.5, linewidth=1.5)
            # Add demographic label on the right
            mid_pos = current_pos + (n_values - 1) / 2
            ax.text(102, mid_pos, demo.replace('_', ' ').title(),
                   ha='left', va='center', fontsize=7, style='italic', rotation=-90,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
            current_pos += n_values
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9, 
             title='Medication Type', framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'medication_distribution_by_model.png'
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
        
        # Generate specialized plots
        print("\n" + "=" * 60)
        print("GENERATING SPECIALIZED PLOTS")
        print("=" * 60)
        
        print("\n📊 Creating surgical referral by model plot...")
        plot_surgical_referral_by_model(baseline, full, output_dir)
        
        print("\n📊 Creating TTD duration by model plot...")
        plot_ttd_duration_by_model(baseline, full, output_dir)
        
        print("\n📊 Creating medication distribution by model plot...")
        plot_medication_distribution_by_model(baseline, full, output_dir)
        
    else:
        print("\n⚠️  No comparison results generated")


if __name__ == "__main__":
    main()

