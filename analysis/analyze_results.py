#!/usr/bin/env python3
"""
Analyze experiment results and compute invasiveness index.

Usage:
    python analyze_results.py results/results_final_20240101_120000.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))
import config


def calculate_invasiveness(row):
    """Calculate invasiveness index (0-10 scale) for a single row."""
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


def summarize_by_attribute(df, attribute, outcome='invasiveness_index'):
    """Summarize outcome by sociodemographic attribute."""
    summary = df.groupby(attribute)[outcome].agg([
        'count', 'mean', 'std', 'min',
        ('25%', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        'max'
    ]).round(3)
    return summary


def test_differences(df, attribute, outcome='invasiveness_index'):
    """Test for significant differences across attribute levels."""
    groups = [group[outcome].dropna().values 
             for name, group in df.groupby(attribute)]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return None
    
    f_stat, p_value = stats.f_oneway(*groups)
    h_stat, p_value_kw = stats.kruskal(*groups)
    
    return {
        "attribute": attribute,
        "n_groups": len(groups),
        "anova_f": f_stat,
        "anova_p": p_value,
        "kruskal_h": h_stat,
        "kruskal_p": p_value_kw,
        "significant": p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('results_file', type=Path, help='Path to results CSV file')
    parser.add_argument('--output', type=Path, help='Output directory for analysis')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: File not found: {args.results_file}")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    df = pd.read_csv(args.results_file)
    
    # Filter successful queries
    df = df[df['success'] == True].copy()
    
    # Exclude baseline models (use compare_to_baseline.py for baseline analysis)
    df = df[~df['model'].str.startswith('baseline-')].copy()
    
    print(f"\nLoaded {len(df)} successful queries")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Vignettes: {df['vignette_id'].nunique()}")
    
    # Calculate invasiveness
    print("\nCalculating invasiveness index...")
    df['invasiveness_index'] = df.apply(calculate_invasiveness, axis=1)
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL INVASIVENESS INDEX")
    print("="*60)
    print(df['invasiveness_index'].describe())
    
    # By model
    print("\n" + "="*60)
    print("INVASIVENESS BY MODEL")
    print("="*60)
    print(summarize_by_attribute(df, 'model'))
    
    # By each sociodemographic attribute
    attributes = ['age_band', 'race_ethnicity', 'gender_identity', 
                  'sexual_orientation', 'socioeconomic_status', 
                  'occupation_type', 'language_proficiency', 'geography']
    
    print("\n" + "="*60)
    print("INVASIVENESS BY SOCIODEMOGRAPHIC ATTRIBUTES")
    print("="*60)
    
    for attr in attributes:
        if attr in df.columns:
            print(f"\n{attr.upper().replace('_', ' ')}:")
            print(summarize_by_attribute(df, attr))
    
    # Statistical tests
    print("\n" + "="*60)
    print("STATISTICAL TESTS FOR DIFFERENCES")
    print("="*60)
    
    test_results = []
    for attr in attributes:
        if attr in df.columns:
            result = test_differences(df, attr)
            if result:
                test_results.append(result)
    
    test_df = pd.DataFrame(test_results)
    print(test_df.to_string(index=False))
    
    # Outcome frequencies
    print("\n" + "="*60)
    print("CATEGORICAL OUTCOME FREQUENCIES")
    print("="*60)
    
    outcomes = ['Medication prescription', 'work_status', 'surgical_referral']
    for outcome in outcomes:
        if outcome in df.columns:
            print(f"\n{outcome}:")
            print(df[outcome].value_counts(normalize=True).mul(100).round(1))
    
    # Save analysis
    if args.output:
        args.output.mkdir(exist_ok=True)
        
        # Save enhanced dataframe
        output_file = args.output / f"analysis_{args.results_file.stem}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved analysis to {output_file}")
        
        # Save summaries
        summary_file = args.output / f"summary_{args.results_file.stem}.txt"
        with open(summary_file, 'w') as f:
            f.write("STATISTICAL TEST RESULTS\n")
            f.write("="*60 + "\n")
            f.write(test_df.to_string(index=False))
        print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()

