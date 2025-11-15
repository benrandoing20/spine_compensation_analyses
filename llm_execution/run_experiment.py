#!/usr/bin/env python3
"""
Run the spine compensation bias experiment.

Usage:
    # Test with 10 vignettes using free NVIDIA models
    python run_experiment.py --test
    
    # Full run with specific models
    python run_experiment.py --models gpt-4o claude-3.5-sonnet llama-3.3-70b
    
    # List available models
    python run_experiment.py --list-models
"""

import argparse
import itertools
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import config
from llm_execution.llm_providers import query_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_vignettes():
    """Generate all possible vignette combinations (full factorial design)."""
    attr_names = list(config.ATTRIBUTES.keys())
    attr_values = [config.ATTRIBUTES[name] for name in attr_names]
    
    vignettes = []
    for idx, combo in enumerate(itertools.product(*attr_values), start=1):
        vignette = {"vignette_id": idx}
        vignette.update(dict(zip(attr_names, combo)))
        vignettes.append(vignette)
    
    return vignettes


def run_single_query(model_key, vignette, replicate):
    """Run a single query and return structured result."""
    # Format prompt
    prompt = config.VIGNETTE_TEMPLATE.format(**vignette)
    
    # Generate seed for reproducibility
    seed = hash(f"{vignette['vignette_id']}_{model_key}_{replicate}") % (2**31)
    
    try:
        result = query_model(
            model_key=model_key,
            prompt=prompt,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            seed=seed
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model": model_key,
            "vignette_id": vignette["vignette_id"],
            "replicate": replicate,
            "seed": seed,
            **vignette,
            **result["parsed_response"],
            "raw_response": result["response"],
            "api_metadata": result["metadata"],
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Query failed for vignette {vignette['vignette_id']}, "
                    f"model {model_key}, replicate {replicate}: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "model": model_key,
            "vignette_id": vignette["vignette_id"],
            "replicate": replicate,
            "seed": seed,
            **vignette,
            "success": False,
            "error": str(e)
        }


def save_results(results, suffix="final", output_dir="analysis/results"):
    """Save results to JSON and CSV."""
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = results_dir / f"results_{suffix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = results_dir / f"results_{suffix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved {len(results)} results to {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run spine compensation bias experiment")
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--max-vignettes', type=int, help='Max vignettes (for testing)')
    parser.add_argument('--vignette-range', type=str, help='Vignette range, e.g. "1-500" or "501-1000"')
    parser.add_argument('--replicates', type=int, default=config.NUM_REPLICATES)
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between queries (seconds)')
    parser.add_argument('--output-dir', type=str, default='analysis/results', help='Output directory for results')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 60)
        for key, cfg in config.MODELS.items():
            print(f"{key:20} {cfg['provider']:15} [{cfg['tier']}]")
        return
    
    # Test mode defaults
    if args.test:
        args.models = args.models or ["llama-3.1-405b"]
        args.max_vignettes = args.max_vignettes or 10
        args.replicates = 2
        logger.info("Running in TEST mode")
    
    # Default models - all available models
    if args.models is None:
        args.models = [
            # "gpt-5-mini",            # OpenAI latest (Aug 2025)
            "gpt-4o",           # OpenAI
            # "gpt-oss-20b",      # NVIDIA
            "llama-3.3-70b",    # Meta Llama (fixed missing comma)
            "llama-3.1-405b",
            "deepseek-v3.1",    # DeepSeek
            "deepseek-r1",
            "qwen3-next-80b",   # Qwen
            # "qwq-32b",
            # "kimi-k2",          # Kimi
            "mistral-medium-3", # Mistral
            "mistral-small-3.1"
        ]
    
    # Validate models
    invalid = [m for m in args.models if m not in config.MODELS]
    if invalid:
        logger.error(f"Invalid models: {invalid}")
        logger.info("Run with --list-models to see available options")
        return
    
    # Generate vignettes
    vignettes = generate_vignettes()
    
    # Apply vignette range filter if specified
    if args.vignette_range:
        try:
            start, end = map(int, args.vignette_range.split('-'))
            vignettes = [v for v in vignettes if start <= v['vignette_id'] <= end]
            logger.info(f"Filtered to vignette range: {start}-{end} ({len(vignettes)} vignettes)")
        except ValueError:
            logger.error("Invalid vignette range format. Use 'start-end', e.g. '1-500'")
            return
    
    if args.max_vignettes:
        vignettes = vignettes[:args.max_vignettes]
    
    total_queries = len(vignettes) * len(args.models) * args.replicates
    
    logger.info(f"Starting experiment:")
    logger.info(f"  Vignettes: {len(vignettes)}")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Replicates: {args.replicates}")
    logger.info(f"  Total queries: {total_queries}")
    
    results = []
    
    try:
        with tqdm(total=total_queries, desc="Running experiment") as pbar:
            for vignette in vignettes:
                for model_key in args.models:
                    for replicate in range(1, args.replicates + 1):
                        result = run_single_query(model_key, vignette, replicate)
                        results.append(result)
                        pbar.update(1)
                        
                        # Save periodically
                        if len(results) % 50 == 0:
                            save_results(results, suffix="partial", output_dir=args.output_dir)
                        
                        # Rate limiting
                        time.sleep(args.delay)
    
    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
    
    finally:
        # Final save
        df = save_results(results, suffix="final", output_dir=args.output_dir)
        
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total queries: {len(results)}")
        logger.info(f"Success rate: {df['success'].mean() * 100:.1f}%")
        
        print("\nSample results:")
        print(df[['vignette_id', 'model', 'Medication prescription', 
                  'work_status', 'surgical_referral']].head())


if __name__ == "__main__":
    main()

