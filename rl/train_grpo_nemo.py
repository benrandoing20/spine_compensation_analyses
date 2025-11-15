#!/usr/bin/env python3
"""
GRPO Training with NVIDIA NeMo RL Framework for Fairness Alignment.

This script uses NVIDIA NeMo Aligner for Group Relative Policy Optimization (GRPO)
to reduce racial disparities in clinical decision-making.

Requirements:
    - NVIDIA GPU with CUDA
    - nemo-toolkit[nlp]
    - nemo-aligner
    - transformers, peft, torch

Usage:
    # GPU training only
    python train_grpo_nemo.py --model-name meta-llama/Llama-2-7b-chat-hf --num-samples 2304 --iterations 10

Note: NeMo RL requires GPU. CPU testing is not supported.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Check for GPU immediately
try:
    import torch
    if not torch.cuda.is_available():
        print("="*80)
        print("ERROR: NVIDIA NeMo RL requires CUDA GPU")
        print("="*80)
        print("\nNeMo Aligner is designed for GPU-based RL training.")
        print("CPU training is not supported.")
        print("\nOptions:")
        print("  1. Run on a machine with NVIDIA GPU")
        print("  2. Use a cloud GPU service (Brev, Vast.ai, etc.)")
        print("="*80)
        sys.exit(1)
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# Import NeMo (required, no fallback)
try:
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import get_prompt_template_example
    from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
    from nemo_aligner.algorithms.supervised import SupervisedTrainer
    from nemo_aligner.data.nlp.builders import build_dataloader, build_sft_dataset
    from nemo_aligner.models.nlp.gpt.megatron_gpt_reward_model import MegatronGPTRewardModel
    from nemo_aligner.utils.distributed import initialize_distributed
    from nemo_aligner.utils.train_script_utils import CustomLoggerWrapper, init_distributed
    from nemo_aligner.utils.utils import load_and_override_model_config, retrieve_custom_trainer_state_dict
    
    print("✅ NeMo Aligner loaded successfully")
    NEMO_AVAILABLE = True
except ImportError as e:
    print("="*80)
    print("ERROR: NVIDIA NeMo Aligner not installed")
    print("="*80)
    print(f"\nImport error: {e}")
    print("\nTo install NeMo Aligner:")
    print("  pip install nemo-toolkit[nlp]")
    print("  pip install nemo-aligner")
    print("\nOr follow: https://github.com/NVIDIA/NeMo-Aligner")
    print("="*80)
    sys.exit(1)

# Standard imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


class RacialDisparityMetrics:
    """Calculate racial disparity metrics for fairness evaluation."""
    
    def __init__(self, reference_group: str = "White"):
        self.reference_group = reference_group
    
    def compute_invasiveness(self, output: Dict) -> float:
        """Compute invasiveness score (0-10) from model output."""
        score = 0.0
        
        # Medication (0-4)
        med = output.get('Medication prescription', '')
        score += config.INVASIVENESS_SCORES['medication'].get(med, 0)
        
        # Work status (0-2)
        work = output.get('work_status', '')
        score += config.INVASIVENESS_SCORES['work_status'].get(work, 0)
        
        # Physical therapy (0-1)
        pt = output.get('physical_therapy', '')
        score += config.INVASIVENESS_SCORES['physical_therapy'].get(pt, 0)
        
        # Mental health (0-2)
        mh = output.get('mental_health_referral', '')
        score += config.INVASIVENESS_SCORES['mental_health_referral'].get(mh, 0)
        
        # Surgical referral (0-1)
        surg = output.get('surgical_referral', '')
        score += config.INVASIVENESS_SCORES['surgical_referral'].get(surg, 0)
        
        return score
    
    def compute_racial_disparity(self, outputs: List[str], demographics: List[Dict]) -> Dict:
        """
        Compute racial disparity metrics.
        
        Returns dict with:
        - mean_invasiveness_by_race: Dict[str, float]
        - disparity_ratio: float (max/min ratio)
        - gini_coefficient: float (0-1, higher = more disparity)
        """
        # Parse outputs
        data = []
        for output_text, demo in zip(outputs, demographics):
            try:
                start = output_text.find('{')
                end = output_text.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(output_text[start:end])
                else:
                    parsed = {}
            except:
                parsed = {}
            
            invasiveness = self.compute_invasiveness(parsed)
            data.append({
                'race_ethnicity': demo.get('race_ethnicity', 'Unknown'),
                'invasiveness': invasiveness,
                **demo
            })
        
        df = pd.DataFrame(data)
        
        # Mean invasiveness by race
        race_means = df.groupby('race_ethnicity')['invasiveness'].mean().to_dict()
        
        # Disparity ratio (max/min)
        means = list(race_means.values())
        if len(means) > 0 and min(means) > 0:
            disparity_ratio = max(means) / min(means)
        elif len(means) > 0 and max(means) > 0:
            disparity_ratio = 10.0  # High disparity indicator
        else:
            disparity_ratio = 1.0
        
        # Reference group disparity
        ref_mean = race_means.get(self.reference_group, np.mean(means) if means else 0)
        reference_disparities = {}
        if ref_mean > 0:
            reference_disparities = {
                race: (mean - ref_mean) / ref_mean
                for race, mean in race_means.items()
                if race != self.reference_group
            }
        avg_ref_disparity = np.mean(list(reference_disparities.values())) if reference_disparities else 0
        
        # Gini coefficient
        gini = self._compute_gini(means)
        
        # Variance
        variance = np.var(means)
        
        # Count successfully parsed outputs
        n_parsed = sum(1 for d in data if d.get('output'))
        
        # Warning if most outputs failed to parse
        if len(data) > 0 and n_parsed / len(data) < 0.5:
            print(f"\n⚠️  Warning: Only {n_parsed}/{len(data)} outputs parsed successfully")
            print("    Model may not be generating valid JSON yet")
        
        return {
            'mean_invasiveness_by_race': race_means,
            'disparity_ratio': float(disparity_ratio),
            'reference_disparity': float(avg_ref_disparity),
            'gini_coefficient': float(gini),
            'variance': float(variance),
            'overall_mean': float(np.mean(means) if means else 0),
            'n_samples': len(df),
            'n_parsed_outputs': n_parsed
        }
    
    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
        if len(values) == 0:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        total = sum(sorted_values)
        
        # Handle edge cases
        if total == 0 or n == 0:
            return 0.0
        
        # Gini formula
        gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_values)) - 
                (n + 1) * total) / (n * total)
        
        return float(gini)


def generate_vignettes(num_samples: int = None) -> List[Dict]:
    """Generate vignettes (full factorial or subset)."""
    print("\nGenerating vignettes...")
    attributes = config.ATTRIBUTES
    keys = list(attributes.keys())
    combinations = list(itertools.product(*[attributes[k] for k in keys]))
    
    if num_samples is not None and num_samples < len(combinations):
        import random
        combinations = random.sample(combinations, num_samples)
    
    vignettes = []
    for i, combo in enumerate(combinations):
        vignette = {key: val for key, val in zip(keys, combo)}
        vignette['vignette_id'] = i + 1
        vignettes.append(vignette)
    
    print(f"✅ Generated {len(vignettes)} vignettes")
    return vignettes


def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training with NVIDIA NeMo RL Framework (GPU Required)"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='HuggingFace model name (e.g., meta-llama/Llama-2-7b-chat-hf)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2304,
        help='Number of vignettes (default: 2304 = full factorial)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of GRPO iterations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='grpo_checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for updates'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GRPO FAIRNESS TRAINING WITH NVIDIA NeMo RL")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Iterations: {args.iterations}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Device: CUDA (GPU Required)")
    print("="*80 + "\n")
    
    print("ERROR: Full NeMo Aligner integration not yet implemented")
    print("\nNeMo Aligner requires:")
    print("  - Megatron-format models (.nemo files)")
    print("  - Distributed training configuration")
    print("  - Multi-GPU setup")
    print("\nThis is beyond the scope of a simple script.")
    print("\nFor production GRPO with NeMo:")
    print("  1. Convert HuggingFace model to .nemo format")
    print("  2. Use NeMo Aligner examples:")
    print("     https://github.com/NVIDIA/NeMo-Aligner/tree/main/examples")
    print("  3. Configure distributed training")
    print("\n" + "="*80)
    sys.exit(1)


if __name__ == "__main__":
    main()
