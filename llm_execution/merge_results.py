#!/usr/bin/env python3
"""
Merge multiple result files from batched runs into a single dataset.

Usage:
    python merge_results.py
    python merge_results.py --output merged_results.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def merge_results(results_dir: Path = Path("analysis/results"), output_name: str = "merged_results", recursive: bool = True):
    """Merge all final result CSVs into one file."""
    
    # Find all final result files (including subdirectories)
    if recursive:
        csv_files = sorted(results_dir.glob("**/results_final_*.csv"))
        print(f"Searching recursively in {results_dir} and subdirectories...")
    else:
        csv_files = sorted(results_dir.glob("results_final_*.csv"))
    
    if not csv_files:
        print(f"❌ No result files found in {results_dir}/")
        return
    
    print(f"Found {len(csv_files)} result files")
    print("=" * 60)
    
    # Show which files/directories are being merged
    for csv_file in csv_files:
        relative_path = csv_file.relative_to(results_dir)
        print(f"  📁 {relative_path}")
    
    # Load and concatenate
    print("\nLoading files...")
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error loading {csv_file.name}: {e}")
    
    # Merge
    merged = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (in case of overlapping runs)
    print(f"\nTotal rows before dedup: {len(merged)}")
    merged = merged.drop_duplicates(
        subset=['model', 'vignette_id', 'replicate'],
        keep='first'
    )
    print(f"Total rows after dedup: {len(merged)}")
    
    # Summary stats
    print("\n" + "=" * 60)
    print("MERGED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total queries: {len(merged)}")
    print(f"Successful: {merged['success'].sum()}")
    print(f"Failed: {(~merged['success']).sum()}")
    print(f"Success rate: {merged['success'].mean() * 100:.1f}%")
    print(f"\nUnique vignettes: {merged['vignette_id'].nunique()}")
    print(f"Models: {merged['model'].nunique()}")
    print(f"Replicates per vignette: {merged.groupby(['model', 'vignette_id']).size().max()}")
    
    print("\nModels in dataset:")
    for model, count in merged['model'].value_counts().items():
        print(f"  {model}: {count} queries")
    
    # Save merged results
    output_csv = results_dir / f"{output_name}.csv"
    output_json = results_dir / f"{output_name}.json"
    
    merged.to_csv(output_csv, index=False)
    print(f"\n✅ Saved merged CSV: {output_csv}")
    
    # Also save as JSON for completeness
    records = merged.to_dict('records')
    with open(output_json, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"✅ Saved merged JSON: {output_json}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge batched experiment results")
    parser.add_argument('--output', default='merged_results', help='Output filename (without extension)')
    parser.add_argument('--results-dir', default='results', help='Directory containing result files')
    parser.add_argument('--no-recursive', action='store_true', help='Do not search subdirectories')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    merge_results(results_dir, args.output, recursive=not args.no_recursive)


if __name__ == "__main__":
    main()

