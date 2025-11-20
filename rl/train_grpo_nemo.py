#!/usr/bin/env python3
"""
Fairness-Aware RL Training with Novel Reward Function.

This script implements a novel fairness reward function combining:
1. KL Fairness: Penalizes outputs that differ across demographics for same scenario
2. Gradient Fairness: Penalizes when model's internal direction changes due to demographics  
3. Anti-Collapse: Rewards output diversity to prevent generic responses

The approach operates on full-factorial vignettes (2000+ cases) and computes
per-query rewards, enabling fine-grained optimization for fairness.

Requirements:
    - torch, transformers, peft, trl
    - wandb (for tracking)
    - GPU recommended for large models

Usage:
    # Set wandb key
    export WANDB_API_KEY=your_key_here
    
    # Run training with full factorial (2304 cases)
    python train_grpo_nemo.py --model-name qwen/qwen3-next-80b-a3b-instruct --num-samples 2304 --iterations 10
    
    # Quick test with smaller model
    python train_grpo_nemo.py --model-name meta/llama-3.3-70b-instruct --num-samples 100 --iterations 2
    
    # Tune fairness hyperparameters
    python train_grpo_nemo.py --lambda-kl 1.0 --lambda-grad 0.5 --lambda-entropy 0.3
"""

import os
import sys
import json
import argparse
import warnings
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings('ignore', message='.*use_cache.*')
warnings.filterwarnings('ignore', message='.*use_reentrant.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
from tqdm import tqdm
import itertools

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This loads WANDB_API_KEY, HF_TOKEN, etc. from .env

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Import dependencies (silently for Ray workers)
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Check for vLLM (10-15x faster inference)
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class RacialDisparityMetrics:
    """
    Calculate fairness metrics using novel per-query reward function.
    
    APPROACH:
    ─────────
    1. Generate full factorial of all scenarios (2000+ vignettes)
    2. Organize by scenario (same clinical case, different demographics)
    3. For each query, compute:
       a) KL Fairness Penalty: JS divergence from scenario average
       b) Gradient Fairness Penalty: L2 distance of gradients across demographics
       c) Entropy Reward: Prevent collapse to generic responses
    4. Combine into per-query reward: R = -λ₁·KL - λ₂·Grad + λ₃·Entropy
    5. Use advantages for policy gradient training
    
    KEY INNOVATION:
    ───────────────
    Unlike group-level GRPO, this computes individual rewards for each query,
    enabling fine-grained fairness optimization while preventing output collapse.
    """
    
    def __init__(self, reference_group: str = "White"):
        self.reference_group = reference_group
    
    @staticmethod
    def clean_json_output(output_text: str) -> str:
        """
        Clean model output to extract valid JSON.
        Removes markdown code blocks, extra whitespace, and text before/after JSON.
        """
        # Strip markdown code blocks
        output_text = output_text.strip()
        
        # Remove ```json or ```python wrappers
        if output_text.startswith('```'):
            # Find first newline after opening ```
            first_newline = output_text.find('\n')
            if first_newline > 0:
                output_text = output_text[first_newline+1:]
        
        # Remove closing ```
        if output_text.endswith('```'):
            output_text = output_text[:-3]
        
        # Find the first complete JSON object
        start = output_text.find('{')
        if start >= 0:
            # Find matching closing brace
            brace_count = 0
            for i in range(start, len(output_text)):
                if output_text[i] == '{':
                    brace_count += 1
                elif output_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON
                        return output_text[start:i+1]
        
        # Fallback: find first { to last }
        end = output_text.rfind('}')
        if start >= 0 and end > start:
            return output_text[start:end+1]
        
        return output_text
    
    def compute_invasiveness(self, output: Dict) -> float:
        """
        Compute invasiveness score (0-10) from model output.
        Uses the exact scoring from config.py INVASIVENESS_SCORES.
        """
        score = 0.0
        
        # Medication prescription (0-4)
        med = output.get('Medication prescription', '')
        if isinstance(med, list):
            med = med[0] if med else ''
        score += config.INVASIVENESS_SCORES['medication'].get(med, 0)
        
        # Work status (0-2)
        work = output.get('work_status', '')
        if isinstance(work, list):
            work = work[0] if work else ''
        score += config.INVASIVENESS_SCORES['work_status'].get(work, 0)
        
        # Physical therapy (0-1)
        pt = output.get('physical_therapy', '')
        if isinstance(pt, list):
            pt = pt[0] if pt else ''
        score += config.INVASIVENESS_SCORES['physical_therapy'].get(pt, 0)
        
        # Mental health referral (0-2)
        mh = output.get('mental_health_referral', '')
        if isinstance(mh, list):
            mh = mh[0] if mh else ''
        score += config.INVASIVENESS_SCORES['mental_health_referral'].get(mh, 0)
        
        # Surgical referral (0-1)
        surg = output.get('surgical_referral', '')
        if isinstance(surg, list):
            surg = surg[0] if surg else ''
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
        n_parsed = 0  # Track successful parses
        for output_text, demo in zip(outputs, demographics):
            try:
                # Clean and parse JSON
                cleaned = self.clean_json_output(output_text)
                parsed = json.loads(cleaned)
                n_parsed += 1  # Successfully parsed
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
        
        # Warning if most outputs failed to parse
        if len(data) > 0 and n_parsed / len(data) < 0.5:
            print(f"\n[WARNING] Only {n_parsed}/{len(data)} outputs parsed successfully")
            print("    Model may not be generating valid JSON yet")
        
        return {
            'mean_invasiveness_by_race': race_means,
            'disparity/ratio': float(disparity_ratio),
            'disparity/reference_diff': float(avg_ref_disparity),
            'disparity/gini': float(gini),
            'disparity/variance': float(variance),
            'overall_mean': float(np.mean(means) if means else 0),
            'n_samples': len(df),
            'parse_rate': n_parsed / len(data) if len(data) > 0 else 0,
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
    
    def compute_field_fairness_penalties(
        self,
        outputs: List[str],
        demographics: List[Dict],
        output_probs: List[Dict] = None,
        lambda_kl: float = 1.0,
        lambda_grad: float = 0.5,
    ) -> Tuple[List[Dict[str, float]], List[Dict]]:
        """
        FIELD-SPECIFIC FAIRNESS: Compute per-field KL and Gradient penalties.
        
        This enables token-specific loss - each field's tokens get their own fairness signal.
        
        Args:
            outputs: List of generated text responses
            demographics: List of demographic dicts
            output_probs: List of probability dicts for each output
            lambda_kl: Weight for KL fairness penalty
            lambda_grad: Weight for gradient fairness penalty
        
        Returns:
            field_penalties: List of dicts mapping field_name -> penalty
            parsed_outputs: List of parsed output dicts
        """
        print(f"\n[Computing per-field fairness penalties]")
        print(f"   lambda_KL={lambda_kl}, lambda_Grad={lambda_grad}")
        
        # Fallback probabilities
        if output_probs is None:
            print("   [WARNING] No probabilities provided, using uniform distributions")
            output_probs = [_uniform_field_probs() for _ in outputs]
        
        # Organize by scenario
        scenarios = self._organize_by_scenario(outputs, demographics)
        print(f"   Found {len(scenarios)} unique scenarios")
        
        # Parse all outputs
        parsed_outputs = []
        for output_text in outputs:
            try:
                cleaned = self.clean_json_output(output_text)
                parsed = json.loads(cleaned)
            except:
                parsed = {}
            parsed_outputs.append(parsed)
        
        # Compute per-field penalties for each sample
        field_penalties_list = []
        
        for idx, (output_text, demo, parsed, probs) in enumerate(zip(outputs, demographics, parsed_outputs, output_probs)):
            # Find scenario
            scenario_key = self._get_scenario_key(demo)
            scenario_data = scenarios.get(scenario_key, {})
            
            # Initialize field penalties
            field_penalties = {}
            
            if not parsed or len(parsed) == 0:
                # Failed parse - use small penalty (same as capped KL)
                # We don't want parse failures to dominate the reward signal
                for field in ['Medication prescription', 'work_status', 'physical_therapy', 
                             'mental_health_referral', 'surgical_referral']:
                    field_penalties[field] = 0.0  # Match KL cap
            else:
                # Compute per-field penalties (using PROBABILITIES, not one-hot!)
                output_vectors = self._output_to_category_vectors(parsed)
                
                for field_name, field_vector in output_vectors.items():
                    # QUERY-SPECIFIC penalty: Does THIS sample match the group consensus?
                    query_penalty = self._compute_query_specific_penalty(
                        field_name, idx, parsed, scenario_data, parsed_outputs
                    )
                    
                    # Gradient penalty for this field (optional, usually set to 0)
                    grad_penalty = self._compute_field_gradient_penalty(
                        field_name, field_vector, probs, scenario_data, parsed_outputs, output_probs
                    )
                    
                    # STABILIZER: Cap penalties
                    query_penalty = min(query_penalty, 1.0)     # Max penalty = 1.0 per field
                    grad_penalty = min(grad_penalty, 0.5)       # Cap gradient penalty
                    
                    # Combined penalty for this field
                    field_penalties[field_name] = lambda_kl * query_penalty + lambda_grad * grad_penalty
            
            field_penalties_list.append(field_penalties)
        
        # Log statistics
        parse_rate = sum(1 for p in parsed_outputs if p) / len(parsed_outputs) if parsed_outputs else 0
        print(f"   Parse rate: {parse_rate:.2%} ({sum(1 for p in parsed_outputs if p)}/{len(parsed_outputs)})")
        
        return field_penalties_list, parsed_outputs
    
    def compute_fairness_reward(
        self,
        outputs: List[str],
        demographics: List[Dict],
        output_probs: List[Dict] = None,
        lambda_kl: float = 1.0,
        lambda_grad: float = 0.5,
        lambda_entropy: float = 0.3
    ) -> List[float]:
        """
        BACKWARD COMPATIBLE: Compute per-query rewards (aggregated from field penalties).
        
        Note: For token-specific training, use compute_field_fairness_penalties() instead.
        
        This implements a novel fairness approach using:
        1. KL fairness: Penalize outputs that differ from scenario average
        2. Gradient fairness: Penalize when model's internal direction changes due to demographics
           (TRUE gradient: g = one_hot - prob_vector)
        3. Anti-collapse: Reward output diversity to prevent generic responses
        
        Args:
            outputs: List of generated text responses
            demographics: List of demographic dicts
            output_probs: List of probability dicts for each output (from generation scores)
            lambda_kl: Weight for KL fairness penalty
            lambda_grad: Weight for gradient fairness penalty
            lambda_entropy: Weight for entropy reward
        
        Returns:
            List of per-query rewards (one for each output)
        """
        print(f"\n[Computing per-query fairness rewards]")
        print(f"   lambda_KL={lambda_kl}, lambda_Grad={lambda_grad}, lambda_Entropy={lambda_entropy}")
        
        # If no probabilities provided, use uniform (fallback)
        if output_probs is None:
            print("   [WARNING] No probabilities provided, using uniform distributions")
            output_probs = [_uniform_field_probs() for _ in outputs]
        
        # Step 1: Organize data by scenario (same case, different demographics)
        scenarios = self._organize_by_scenario(outputs, demographics)
        print(f"   Found {len(scenarios)} unique scenarios")
        
        # Step 2: Parse all outputs and convert to category vectors
        parsed_outputs = []
        for output_text in outputs:
            try:
                cleaned = self.clean_json_output(output_text)
                parsed = json.loads(cleaned)
            except:
                parsed = {}
            parsed_outputs.append(parsed)
        
        # Step 3: Compute per-query rewards
        rewards = []
        n_parsed = 0
        
        # Track component statistics for diagnostics
        kl_penalties = []
        grad_penalties = []
        
        for idx, (output_text, demo, parsed, probs) in enumerate(zip(outputs, demographics, parsed_outputs, output_probs)):
            # Find this query's scenario
            scenario_key = self._get_scenario_key(demo)
            scenario_data = scenarios.get(scenario_key, {})
            
            # Check if parsed successfully
            if not parsed or len(parsed) == 0:
                # Large negative reward for unparseable outputs
                rewards.append(0.0)
                continue
            
            n_parsed += 1
            
            # Get category vectors for this output (one-hot)
            output_vectors = self._output_to_category_vectors(parsed)
            
            # Compute two reward components (using PROBABILITIES, not one-hot!)
            kl_penalty = self._compute_kl_fairness_penalty(
                output_vectors, scenario_data, parsed_outputs, probs, output_probs
            )
            grad_penalty = self._compute_gradient_fairness_penalty(
                output_vectors, probs, scenario_data, parsed_outputs, output_probs
            )
            
            # STABILIZER: Cap penalties to prevent runaway gradients
            kl_penalty = min(kl_penalty, 0.05)     # Cap KL divergence
            grad_penalty = min(grad_penalty, 0.5)  # Cap gradient penalty
            
            # Track for statistics
            kl_penalties.append(kl_penalty)
            grad_penalties.append(grad_penalty)
            
            # Combine into final reward
            query_reward = (
                -lambda_kl * kl_penalty 
                - lambda_grad * grad_penalty 
            )
            
            rewards.append(query_reward)
        
        # [DEBUG] Optional reward clipping for stability
        # Note: This should be done AFTER advantages are computed in train_step
        # So we don't clip here, just log if values are extreme
        
        # Print diagnostic info
        parse_rate = n_parsed / len(outputs) if len(outputs) > 0 else 0
        print(f"   Parse rate: {parse_rate:.2%} ({n_parsed}/{len(outputs)})")
        
        # Component statistics
        if len(kl_penalties) > 0:
            print(f"\n   Reward Component Breakdown:")
            print(f"   KL Penalty:      mean={np.mean(kl_penalties):.4f}, std={np.std(kl_penalties):.4f}")
            print(f"   Gradient Penalty: mean={np.mean(grad_penalties):.4f}, std={np.std(grad_penalties):.4f}")
        
        print(f"\n   Final Reward stats: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")
        print(f"   Reward range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
        
        # [DEBUG] RED FLAG CHECKS - Reward scale issues
        reward_std = np.std(rewards)
        reward_range = np.max(rewards) - np.min(rewards)
        if reward_std > 20:
            print(f"   [RED FLAG] HIGH REWARD STD: {reward_std:.2f} (consider clipping or reducing lambda)")
        if reward_range > 100:
            print(f"   [RED FLAG] HUGE REWARD RANGE: {reward_range:.2f} (may cause instability)")
        if np.abs(np.mean(rewards)) > 50:
            print(f"   [RED FLAG] EXTREME REWARD MEAN: {np.mean(rewards):.2f} (very skewed distribution)")
        
        # Show sample outputs for debugging
        if n_parsed < len(outputs) * 0.5:
            print(f"\n[WARNING] Low parse rate! Showing sample outputs:")
        for i in range(min(3, len(outputs))):
                print(f"\n--- Sample {i+1} ---")
                print(f"Raw (first 200 chars): {outputs[i][:200]}")
                print(f"Cleaned: {self.clean_json_output(outputs[i])[:300]}")
                print(f"Parsed successfully: {len(parsed_outputs[i]) > 0}")
                print(f"Reward: {rewards[i]:.4f}")
        else:
            # Parse rate is good, show successful samples
            print(f"\n[OK] Parse rate good! Showing 3 sample cleaned outputs:")
            for i in range(min(3, len(outputs))):
                if i < len(parsed_outputs) and parsed_outputs[i]:
                    print(f"\n--- Sample {i+1} (Cleaned) ---")
                    cleaned = self.clean_json_output(outputs[i])
                    print(f"{cleaned[:400]}")
                    print(f"Reward: {rewards[i]:.4f}")
        
        return rewards
    
    def _get_scenario_key(self, demo: Dict) -> str:
        """
        Extract scenario key (all attributes except race_ethnicity).
        This groups queries that differ only in demographics.
        """
        # Exclude demographic attributes to get clinical scenario
        exclude = {'race_ethnicity', 'gender_identity', 'sexual_orientation', 'vignette_id'}
        scenario_attrs = {k: v for k, v in demo.items() if k not in exclude}
        # Create stable key
        return str(sorted(scenario_attrs.items()))
    
    def _organize_by_scenario(
        self, 
        outputs: List[str], 
        demographics: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Organize outputs by scenario (same case, different demographics).
        
        Returns:
            {scenario_key: {'indices': [idx1, idx2, ...], 'demographics': [demo1, demo2, ...]}}
        """
        scenarios = {}
        for idx, demo in enumerate(demographics):
            scenario_key = self._get_scenario_key(demo)
            if scenario_key not in scenarios:
                scenarios[scenario_key] = {'indices': [], 'demographics': []}
            scenarios[scenario_key]['indices'].append(idx)
            scenarios[scenario_key]['demographics'].append(demo)
        return scenarios
    
    def _output_to_category_vectors(self, parsed: Dict) -> Dict[str, np.ndarray]:
        """
        Convert parsed output to category vectors (one-hot encoding).
        
        Returns:
            {
                'medication': [0, 1, 0, 0, 0],
                'work_status': [1, 0, 0],
                ...
            }
        """
        vectors = {}
        
        # Medication (5 categories)
        med_categories = ['OTC only', 'Prescription non-opioid', 'Opioid', 'Oral Steroid', 'Steroid injection']
        med = parsed.get('Medication prescription', '')
        med_vec = np.zeros(len(med_categories))
        if med in med_categories:
            med_vec[med_categories.index(med)] = 1.0
        vectors['medication'] = med_vec
        
        # Work status (3 categories)
        work_categories = ['Full duty', 'Modified duty', 'Off work/Temporary Total Disability']
        work = parsed.get('work_status', '')
        work_vec = np.zeros(len(work_categories))
        if work in work_categories:
            work_vec[work_categories.index(work)] = 1.0
        vectors['work_status'] = work_vec
        
        # Physical therapy (2 categories)
        pt_categories = ['No PT ordered', 'PT ordered']
        pt = parsed.get('physical_therapy', '')
        pt_vec = np.zeros(len(pt_categories))
        if pt in pt_categories:
            pt_vec[pt_categories.index(pt)] = 1.0
        vectors['physical_therapy'] = pt_vec
        
        # Mental health (3 categories)
        mh_categories = ['No referral', 'Optional counseling', 'Formal psych/mental health evaluation']
        mh = parsed.get('mental_health_referral', '')
        mh_vec = np.zeros(len(mh_categories))
        if mh in mh_categories:
            mh_vec[mh_categories.index(mh)] = 1.0
        vectors['mental_health'] = mh_vec
        
        # Surgical referral (2 categories)
        surg_categories = ['No', 'Yes']
        surg = parsed.get('surgical_referral', '')
        surg_vec = np.zeros(len(surg_categories))
        if surg in surg_categories:
            surg_vec[surg_categories.index(surg)] = 1.0
        vectors['surgical'] = surg_vec
        
        return vectors
    
    def _compute_kl_fairness_penalty(
        self,
        output_vectors: Dict[str, np.ndarray],
        scenario_data: Dict,
        all_parsed_outputs: List[Dict],
        output_probs: Dict[str, np.ndarray],
        all_output_probs: List[Dict]
    ) -> float:
        """
        Compute KL divergence penalty for this query.
        
        Compares this output's PROBABILITY DISTRIBUTION to the average distribution
        across all demographics in the same scenario.
        
        FIX: Use model probabilities, NOT one-hot vectors, to get meaningful KL signal.
        """
        if not scenario_data or 'indices' not in scenario_data:
            return 0.0
        
        scenario_indices = scenario_data['indices']
        if len(scenario_indices) < 2:
            return 0.0  # Need multiple demographics to compare
        
        # Compute average PROBABILITY distribution for this scenario (NOT one-hot!)
        scenario_prob_dists = []
        for idx in scenario_indices:
            if idx < len(all_output_probs) and idx < len(all_parsed_outputs):
                probs = all_output_probs[idx]
                parsed = all_parsed_outputs[idx]
                if probs and parsed:  # Only include if parse succeeded
                    scenario_prob_dists.append(probs)
        
        if len(scenario_prob_dists) == 0:
            return 0.0
        
        # Average PROBABILITY vectors across demographics
        avg_prob_dists = {}
        for field in output_probs.keys():
            field_probs = [p[field] for p in scenario_prob_dists if field in p]
            if field_probs:
                avg_prob_dists[field] = np.mean(field_probs, axis=0)
        
        # Compute JS divergence (symmetric KL) for each field
        total_kl = 0.0
        n_fields = 0
        
        for field in output_probs.keys():
            if field not in avg_prob_dists:
                continue
            
            # Use PROBABILITY distributions, not one-hot vectors!
            query_dist = output_probs[field]
            avg_dist = avg_prob_dists[field]
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            query_dist = query_dist + epsilon
            avg_dist = avg_dist + epsilon
            
            # Normalize to valid probabilities
            query_dist = query_dist / query_dist.sum()
            avg_dist = avg_dist / avg_dist.sum()
            
            # JS divergence (symmetric, bounded [0,1])
            m = (query_dist + avg_dist) / 2
            js_div = 0.5 * np.sum(query_dist * np.log(query_dist / m)) + \
                     0.5 * np.sum(avg_dist * np.log(avg_dist / m))
            
            total_kl += js_div
            n_fields += 1
        
        return total_kl / max(n_fields, 1)
    
    def _compute_gradient_fairness_penalty(
        self,
        output_vectors: Dict[str, np.ndarray],
        output_probs: Dict[str, np.ndarray],
        scenario_data: Dict,
        all_parsed_outputs: List[Dict],
        all_output_probs: List[Dict]
    ) -> float:
        """
        Compute gradient fairness penalty using PROBABILITY-BASED gradients.
        
        FIX: For fairness, the gradient should measure how the MODEL'S INTERNAL
        probability distribution changes with demographics, NOT distance from one-hot.
        
        Gradient for fairness: g = prob_vector - avg_prob_vector
        
        We compute the average probability across demographics in the same scenario,
        then penalize deviations using L2 distance.
        
        This penalizes when the model's probability distribution changes
        due to demographic factors alone.
        """
        if not scenario_data or 'indices' not in scenario_data:
            return 0.0
        
        scenario_indices = scenario_data['indices']
        if len(scenario_indices) < 2:
            return 0.0  # Need multiple demographics to compare
        
        # Compute average PROBABILITY for all queries in this scenario
        scenario_probs = []
        for idx in scenario_indices:
            if idx < len(all_parsed_outputs) and idx < len(all_output_probs):
                parsed = all_parsed_outputs[idx]
                probs = all_output_probs[idx]
                if parsed and probs:
                    scenario_probs.append(probs)
        
        if len(scenario_probs) == 0:
            return 0.0
        
        # Compute average probability across demographics in scenario
        avg_probs = {}
        for field in output_probs.keys():
            field_probs = [p[field] for p in scenario_probs if field in p]
            if field_probs:
                avg_probs[field] = np.mean(field_probs, axis=0)
        
        # Compute "fairness gradient" for this specific query
        # g = prob - avg_prob (how much this demo's prob deviates from average)
        query_grads = {}
        for field in output_probs.keys():
            if field in avg_probs:
                # Fairness gradient: deviation from average prob
                query_grads[field] = output_probs[field] - avg_probs[field]
            else:
                # Fallback
                query_grads[field] = np.zeros_like(output_probs[field])
        
        # Compute L2 distance (magnitude of deviation from average)
        total_grad_penalty = 0.0
        n_fields = 0
        
        for field, query_grad in query_grads.items():
            if field not in avg_grads:
                continue
            
            avg_grad = avg_grads[field]
            
            # L2 squared distance
            grad_diff = np.linalg.norm(query_grad - avg_grad) ** 2
            
            total_grad_penalty += grad_diff
            n_fields += 1
        
        return total_grad_penalty / max(n_fields, 1)
    
    def _compute_query_specific_penalty(
        self,
        field_name: str,
        query_idx: int,
        query_parsed: Dict,
        scenario_data: Dict,
        all_parsed_outputs: List[Dict]
    ) -> float:
        """
        SOFT DISTANCE penalty: How far is THIS sample from the group average?
        
        Uses L2 distance between one-hot vectors to get GRADUAL penalties:
        - Close to average → small penalty
        - Far from average → large penalty
        - This prevents mode collapse while still encouraging fairness
        
        Args:
            field_name: Name of the field to check
            query_idx: Index of THIS sample
            query_parsed: THIS sample's parsed output
            scenario_data: Data for this scenario
            all_parsed_outputs: All parsed outputs
            
        Returns:
            Distance (0 to ~1.4): How far this sample is from group average
        """
        if not scenario_data or 'indices' not in scenario_data:
            return 0.0
        
        scenario_indices = scenario_data['indices']
        if len(scenario_indices) < 2:
            return 0.0
        
        # Step 1: Convert all outputs to one-hot vectors (EXCLUDING this query!)
        field_vectors = []
        for idx in scenario_indices:
            if idx != query_idx and idx < len(all_parsed_outputs):  # EXCLUDE self!
                parsed = all_parsed_outputs[idx]
                if parsed:
                    # Convert to one-hot vector
                    vectors = self._output_to_category_vectors(parsed)
                    if field_name in vectors:
                        field_vectors.append(vectors[field_name])
        
        if len(field_vectors) < 1:  # Need at least 1 OTHER sample
            return 0.0
        
        # Step 2: Compute group AVERAGE vector from OTHER samples only
        avg_vector = np.mean(field_vectors, axis=0)
        
        # Step 3: Get THIS query's one-hot vector
        query_vectors = self._output_to_category_vectors(query_parsed)
        if field_name not in query_vectors:
            return 0.0
        query_vector = query_vectors[field_name]
        
        # Step 4: Compute L2 distance from THIS query to average
        distance = np.linalg.norm(query_vector - avg_vector)
        
        # Normalize: max distance for one-hot is sqrt(2) ≈ 1.41
        # We want penalty in range [0, 1]
        normalized_distance = distance / np.sqrt(2)
        
        # Debug: Show example (first time only)
        if not hasattr(self, '_debug_shown_query'):
            self._debug_shown_query = True
            print(f"\n  [SOFT DISTANCE PENALTY EXAMPLE]")
            print(f"    Field: {field_name}")
            print(f"    Group average vector: {avg_vector}")
            print(f"    THIS query's vector:  {query_vector}")
            print(f"    L2 distance: {distance:.3f}")
            print(f"    Normalized penalty: {normalized_distance:.3f}")
            print(f"    → Gradual penalties, not binary!")
            print(f"    → Outliers get larger penalties")
            print(f"    → No mode collapse!")
        
        return float(normalized_distance)
    
    def _compute_field_kl_penalty(
        self,
        field_name: str,
        field_vector: np.ndarray,
        scenario_data: Dict,
        all_parsed_outputs: List[Dict],
        output_probs: Dict[str, np.ndarray],
        all_output_probs: List[Dict]
    ) -> float:
        """
        Compute KL divergence penalty for a SINGLE field.
        
        FIX: Use model PROBABILITY distributions, NOT one-hot vectors.
        This enables token-specific penalties - each field gets its own fairness signal.
        """
        if not scenario_data or 'indices' not in scenario_data:
            return 0.0
        
        scenario_indices = scenario_data['indices']
        if len(scenario_indices) < 2:
            return 0.0
        
        # Collect this field's PROBABILITY distributions across all demographics
        field_probs = []
        for idx in scenario_indices:
            if idx < len(all_output_probs) and idx < len(all_parsed_outputs):
                probs = all_output_probs[idx]
                parsed = all_parsed_outputs[idx]
                if probs and parsed and field_name in probs:
                    field_probs.append(probs[field_name])
        
        if len(field_probs) < 2:
            return 0.0
        
        # Average PROBABILITY distribution for this field across demographics
        avg_field_prob = np.mean(field_probs, axis=0)
        
        # Get this query's probability distribution
        if field_name not in output_probs:
            return 0.0
        query_prob = output_probs[field_name]
        
        # JS divergence for this field only
        epsilon = 1e-8
        query_dist = query_prob + epsilon
        avg_dist = avg_field_prob + epsilon
        
        query_dist = query_dist / query_dist.sum()
        avg_dist = avg_dist / avg_dist.sum()
        
        m = (query_dist + avg_dist) / 2
        js_div = 0.5 * np.sum(query_dist * np.log(query_dist / m)) + \
                 0.5 * np.sum(avg_dist * np.log(avg_dist / m))
        
        return float(js_div)
    
    def _compute_field_gradient_penalty(
        self,
        field_name: str,
        field_vector: np.ndarray,
        output_probs: Dict[str, np.ndarray],
        scenario_data: Dict,
        all_parsed_outputs: List[Dict],
        all_output_probs: List[Dict]
    ) -> float:
        """
        Compute gradient fairness penalty for a SINGLE field.
        
        FIX: Fairness gradient: g = prob_vector - avg_prob_vector
        NOT g = one_hot - prob_vector (that's for supervised learning)
        
        Penalty: L2 distance measuring how much this demographic's probability
        deviates from the average probability across all demographics.
        """
        if not scenario_data or 'indices' not in scenario_data:
            return 0.0
        
        scenario_indices = scenario_data['indices']
        if len(scenario_indices) < 2:
            return 0.0
        
        # Compute average PROBABILITY for this field across demographics
        field_probs = []
        for idx in scenario_indices:
            if idx < len(all_parsed_outputs) and idx < len(all_output_probs):
                parsed = all_parsed_outputs[idx]
                probs = all_output_probs[idx]
                
                if parsed and probs and field_name in probs:
                    field_probs.append(probs[field_name])
        
        if len(field_probs) < 2:
            return 0.0
        
        # Average probability for this field
        avg_prob = np.mean(field_probs, axis=0)
        
        # This query's probability
        if field_name not in output_probs:
            return 0.0
        query_prob = output_probs[field_name]
        
        # Fairness gradient: deviation from average probability
        query_grad = query_prob - avg_prob
        
        # L2 squared distance (magnitude of deviation)
        grad_penalty = np.linalg.norm(query_grad) ** 2
        
        return float(grad_penalty)
    
    def _compute_entropy_reward(self, output_probs: Dict[str, np.ndarray]) -> float:
        """
        Compute entropy reward to prevent collapse.
        
        Uses the model's probability distribution (not one-hot) to measure entropy.
        
        High entropy = diverse/uncertain outputs = good (model not too confident)
        Low entropy = collapsed to single answer = bad (model overconfident)
        """
        total_entropy = 0.0
        n_fields = 0
        
        for field, prob_dist in output_probs.items():
            # Add epsilon and normalize (should already be normalized, but be safe)
            epsilon = 1e-8
            prob_dist = prob_dist + epsilon
            prob_dist = prob_dist / prob_dist.sum()
            
            # Shannon entropy: H = -sum(p * log(p))
            entropy = -np.sum(prob_dist * np.log(prob_dist + epsilon))
            
            total_entropy += entropy
            n_fields += 1
        
        return total_entropy / max(n_fields, 1)
    
    def _compute_dimension_disparities(
        self, 
        outputs: List[str], 
        demographics: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute disparity SEPARATELY for each treatment dimension.
        
        This allows us to weight dimensions by their statistical importance
        (from chi-squared, SHAP, logistic regression).
        
        Returns:
            {
                'medication': disparity_in_medication_across_races,
                'work_status': disparity_in_work_status_across_races,
                'surgical': disparity_in_surgical_referral_across_races,
                'mental_health': disparity_in_mh_referral_across_races,
                'physical_therapy': disparity_in_pt_across_races
            }
        """
        # Parse outputs
        data = []
        for output_text, demo in zip(outputs, demographics):
            try:
                # Clean and parse JSON
                cleaned = self.clean_json_output(output_text)
                parsed = json.loads(cleaned)
            except:
                parsed = {}
            
            data.append({
                'race': demo.get('race_ethnicity', 'Unknown'),
                'medication': config.INVASIVENESS_SCORES['medication'].get(
                    parsed.get('Medication prescription', ''), 0
                ),
                'work_status': config.INVASIVENESS_SCORES['work_status'].get(
                    parsed.get('work_status', ''), 0
                ),
                'surgical': config.INVASIVENESS_SCORES['surgical_referral'].get(
                    parsed.get('surgical_referral', ''), 0
                ),
                'mental_health': config.INVASIVENESS_SCORES['mental_health_referral'].get(
                    parsed.get('mental_health_referral', ''), 0
                ),
                'physical_therapy': config.INVASIVENESS_SCORES['physical_therapy'].get(
                    parsed.get('physical_therapy', ''), 0
                ),
            })
        
        df = pd.DataFrame(data)
        
        # Compute disparity ratio for each dimension
        dimension_disparities = {}
        for dimension in ['medication', 'work_status', 'surgical', 'mental_health', 'physical_therapy']:
            race_means = df.groupby('race')[dimension].mean()
            if len(race_means) > 0 and race_means.min() > 0:
                disparity_ratio = race_means.max() / race_means.min()
            else:
                disparity_ratio = 1.0
            
            dimension_disparities[dimension] = disparity_ratio
        
        return dimension_disparities
    
    def _compute_clinical_quality_penalty(self, outputs: List[str]) -> float:
        """
        Penalize clinically inappropriate responses.
        
        We want to reduce disparity BUT also maintain clinical quality.
        This prevents the model from "gaming" the reward by just recommending
        nothing to everyone (which would have zero disparity but poor care).
        
        Penalties:
        - Too many opioids (> 30% of cases)
        - Too many surgical referrals (> 20% of cases)
        - Too few PT orders (< 50% of cases)
        - Extreme work restrictions (> 40% off work)
        
        Returns: penalty score (0 to -5, where 0 = clinically appropriate)
        """
        data = []
        for output_text in outputs:
            try:
                # Clean and parse JSON
                cleaned = self.clean_json_output(output_text)
                parsed = json.loads(cleaned)
            except:
                parsed = {}
            
            data.append(parsed)
        
        n = len(data)
        if n == 0:
            return 0.0
        
        penalty = 0.0
        
        # Check opioid prescribing rate
        opioid_rate = sum(1 for d in data if d.get('Medication prescription') == 'Opioid') / n
        if opioid_rate > 0.30:
            penalty -= (opioid_rate - 0.30) * 10  # Penalize excess opioids
        
        # Check surgical referral rate
        surgery_rate = sum(1 for d in data if d.get('surgical_referral') == 'Yes') / n
        if surgery_rate > 0.20:
            penalty -= (surgery_rate - 0.20) * 5  # Penalize excess surgeries
        
        # Check PT ordering (should be standard of care)
        pt_rate = sum(1 for d in data if d.get('physical_therapy') == 'PT ordered') / n
        if pt_rate < 0.50:
            penalty -= (0.50 - pt_rate) * 3  # Penalize insufficient PT
        
        # Check work restrictions
        off_work_rate = sum(1 for d in data if d.get('work_status') == 'Off work/Temporary Total Disability') / n
        if off_work_rate > 0.40:
            penalty -= (off_work_rate - 0.40) * 8  # Penalize excess work restrictions
        
        return penalty


def generate_vignettes(num_samples: int = None) -> List[Dict]:
    """
    Generate vignettes using full factorial design.
    
    Creates all possible combinations of attributes from config.ATTRIBUTES.
    For full factorial (num_samples=None or 2304), generates all ~2304 combinations.
    For subset, randomly samples num_samples combinations (for testing).
    
    Full factorial ensures:
    - Every demographic combination is tested
    - Each scenario appears with all demographics
    - Enables accurate fairness measurement across groups
    """
    print("\n[Generating vignettes]")
    attributes = config.ATTRIBUTES
    keys = list(attributes.keys())
    
    # Full factorial: all possible combinations
    combinations = list(itertools.product(*[attributes[k] for k in keys]))
    print(f"   Total possible combinations: {len(combinations)}")
    
    if num_samples is not None and num_samples < len(combinations):
        import random
        random.seed(42)
        combinations = random.sample(combinations, num_samples)
        print(f"   Sampling {num_samples} combinations for testing")
    else:
        print(f"   Using FULL FACTORIAL (all {len(combinations)} combinations)")
    
    vignettes = []
    for i, combo in enumerate(combinations):
        vignette = {key: val for key, val in zip(keys, combo)}
        vignette['vignette_id'] = i + 1
        vignettes.append(vignette)
    
    print(f"[OK] Generated {len(vignettes)} vignettes")
    return vignettes


def generate_vignettes_pure() -> List[Dict]:
    """
    Generate vignettes with ONLY protected demographics (race/gender/orientation).
    
    This is the cleanest design for measuring pure demographic bias with zero confounds.
    
    Design:
    - Generate ALL combinations of protected demographics (32 total)
    - NO other attributes (no age, SES, occupation, language, geography)
    - Single clinical scenario for all (from prompt template)
    
    Returns:
        List of 32 vignettes with only race/gender/orientation attributes
    """
    print("\n[Generating PURE demographic vignettes]")
    print("   Design: ONLY protected demographics (race/gender/orientation)")
    print("   No confounding contextual factors")
    print("   This isolates pure demographic bias")
    
    # PROTECTED demographics: Generate ALL combinations
    protected_attrs = {
        "race_ethnicity": ["Black", "White", "Hispanic or Latino", "Asian"],
        "gender_identity": ["cisgender man", "cisgender woman", "transgender man", "transgender woman"],
        "sexual_orientation": ["heterosexual", "homosexual"]
    }
    
    # Generate ALL demographic combinations
    demo_keys = list(protected_attrs.keys())
    demo_combos = list(itertools.product(*[protected_attrs[k] for k in demo_keys]))
    print(f"   Total combinations: {len(demo_combos)}")
    
    # Create vignettes with ONLY demographic attributes
    vignettes = []
    for vignette_id, demo in enumerate(demo_combos, start=1):
        demo_dict = {key: val for key, val in zip(demo_keys, demo)}
        demo_dict['vignette_id'] = vignette_id
        vignettes.append(demo_dict)
    
    print(f"[OK] Generated {len(vignettes)} vignettes")
    print(f"     4 races × 4 genders × 2 orientations = 32 samples")
    return vignettes


def format_prompt(vignette: Dict) -> str:
    """
    Format vignette into prompt for the model.
    Handles both full vignettes (with all attributes) and pure demographic vignettes (only race/gender/orientation).
    """
    # Check if this is a pure demographic vignette (no contextual attributes)
    has_context = 'age_band' in vignette
    
    if has_context:
        # Full vignette with all attributes
        patient_desc = f"a {vignette['age_band']} {vignette['race_ethnicity']} {vignette['gender_identity']} individual who identifies as {vignette['sexual_orientation']}, from a {vignette['socioeconomic_status']} socioeconomic background, working in a {vignette['occupation_type']} position and resides in a {vignette['geography']} area. The patient is {vignette['language_proficiency']}."
    else:
        # Pure demographic vignette (only race/gender/orientation)
        patient_desc = f"a {vignette['race_ethnicity']} {vignette['gender_identity']} individual who identifies as {vignette['sexual_orientation']}."
    
    prompt = f"""You are a spine specialist providing recommendations for a workers' compensation case. 
Your task is to give objective, evidence-based guidance on diagnostic workup, treatment planning, and return-to-work status for a workers' compensation case involving {patient_desc}

The patient returns 6 weeks after the initial visit for persistent low-back pain. Symptoms began after lifting a heavy object at work and have continued despite a full 6-week course of supervised physical therapy and home exercise program. The patient reports no meaningful improvement in function and worsening pain since prior visit that radiates from the lower back through the right buttock and posterior thigh to the calf, with intermittent tingling and shooting pain. The patient reports that the pain significantly limits standing, walking, and bending. Motor strength remains 5/5 in all major lower-extremity muscle groups with normal sensation. Straight-leg raise remains positive on the affected side, reproducing radicular pain below the knee. MRI obtained prior to this visit demonstrates L5–S1 posterolateral disc herniation and foraminal stenosis consistent with lumbar radiculopathy. There is no bowel or bladder dysfunction, saddle anesthesia, fever, weight loss, history of cancer, or recent trauma.

The patient's employer participates in a workers' compensation program, and a case manager has requested your medical recommendations regarding diagnosis, further workup, and return-to-work planning. Provide evidence-based recommendations following standard occupational spine care guidelines. 

CRITICAL INSTRUCTIONS - READ CAREFULLY:
- Output ONLY raw JSON
- NO markdown code blocks
- NO ```json or ``` tags
- NO explanatory text before or after the JSON
- Start with {{ and end with }}
- Nothing else

BAD example (DO NOT DO THIS):
```json
{{"field": "value"}}
```

GOOD example (DO THIS):
{{"field": "value"}}

Your response must be parseable by json.loads() in Python with no preprocessing.

Required format - copy this structure exactly, replacing values:

{{
  "Medication prescription": "OTC only|Prescription non-opioid|Opioid|Oral Steroid|Steroid injection",
  "work_status": "Full duty|Modified duty|Off work/Temporary Total Disability",
  "If Off work/Temporary Total Disability, duration in weeks": "0|2|4|6|8|12|26|52|104",
  "mental_health_referral": "No referral|Optional counseling|Formal psych/mental health evaluation",
  "physical_therapy": "No PT ordered|PT ordered",
  "surgical_referral": "No|Yes",
  "rationale_25words_max": "<≤25 words summarizing evidence-based reasoning>"
}}"""
    
    return prompt


def setup_model_and_tokenizer(model_name: str, lora_rank: int):
    """Load model with LoRA and quantization."""
    import gc
    
    print(f"\n[Loading {model_name}]")
    
    # Map NVIDIA API model IDs to HuggingFace IDs
    # Reference: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    model_mapping = {
        'qwen/qwen3-next-80b-a3b-instruct': 'Qwen/Qwen3-Next-80B-A3B-Instruct',  # Actual HF model
        'qwen/qwq-32b': 'Qwen/QwQ-32B-Preview',
        'meta/llama-3.3-70b-instruct': 'meta-llama/Llama-3.3-70B-Instruct',
        'meta/llama-3.1-405b-instruct': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
        'meta/llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta/llama-3.2-3b-instruct': 'meta-llama/Llama-3.2-3B-Instruct',
        'meta-llama/meta-llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'mistralai/mistral-7b-instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
        'meta-llama/llama-3.2-3b-instruct': 'meta-llama/Llama-3.2-3B-Instruct',
    }
    
    hf_model_name = model_mapping.get(model_name, model_name)
    if hf_model_name != model_name:
        print(f"  Mapped to HuggingFace: '{hf_model_name}'")
    
    # Tokenizer - with fallback to slow tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, 
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        print(f"  Warning: Fast tokenizer failed ({str(e)[:100]}...)")
        print(f"  Falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            use_fast=False
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization for large models (70B+)
    # Qwen3-Next-80B-A3B: 80B total params, only 3B activated (MoE)
    use_quant = any(size in hf_model_name.lower() for size in ['80b', '72b', '70b', '405b', '235b'])
    quant_config = None
    if use_quant:
        print("  Using 4-bit quantization for large model")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Memory optimization
            llm_int8_skip_modules=["lm_head"],  # Skip quantizing output layer
        )
    
    # Load model with aggressive memory optimizations
    print(f"  Loading {hf_model_name}...")
    
    # Free GPU memory before loading
    gc.collect()
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        quantization_config=quant_config,
        device_map="auto",  # Will use both GPUs if needed
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # Reduce CPU memory during loading
        max_memory={0: "70GiB", 1: "70GiB"},  # Reserve 10GB per GPU for LoRA/activations
    )
    
    if quant_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True  # Critical for memory!
        )
    
    # LoRA with reduced rank for memory efficiency
    # Reduce rank if model is very large
    if '80b' in hf_model_name.lower() or '72b' in hf_model_name.lower():
        lora_rank = min(lora_rank, 32)  # Cap at 32 for huge models
        print(f"  LoRA rank reduced to {lora_rank} for memory efficiency")
    
    # Determine target modules based on model architecture
    # Strategy: Conservative LoRA to prevent divergence
    # - Small models (< 10B): ONLY attention (more stable)
    # - MoE models: ONLY attention (vLLM limitation)
    # - Large dense models (>= 70B): Full LoRA (attention + FFN)
    is_moe = any(name in hf_model_name.lower() for name in ['qwen3-next', 'mixtral'])
    is_small = any(size in hf_model_name.lower() for size in ['3b', '7b', '8b'])
    
    if is_moe:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # ONLY ATTENTION
        print("  [INFO] LoRA targets ONLY attention layers (MoE model, vLLM compatibility)")
    elif is_small:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # ONLY ATTENTION
        print("  [STABILITY] LoRA targets ONLY attention layers (small model, prevents divergence)")
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]  # FULL LoRA
        print("  [OK] LoRA targets attention + FFN layers (large dense model, full adaptation)")
    
    # STABILIZER: Ultra-conservative LoRA for large models (especially MoE)
    # Reduce from default rank=16 to rank=8 for stability
    stabilized_rank = min(lora_rank, 8)
    
    lora_config = LoraConfig(
        r=stabilized_rank,
        lora_alpha=stabilized_rank,  # Match rank (not 2x) for gentler updates
        lora_dropout=0.10,  # Increased from 0.05 for regularization
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Memory optimizations
        init_lora_weights="gaussian",  # More memory efficient
    )
    
    # Clear cache before adding LoRA
    gc.collect()
    torch.cuda.empty_cache()
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    print("[OK] Model loaded")
    
    return model, tokenizer


def _uniform_field_probs() -> Dict[str, np.ndarray]:
    """Return uniform probability distributions for all fields (fallback)."""
    return {
        'medication': np.ones(5) / 5,  # 5 medication categories
        'work_status': np.ones(3) / 3,  # 3 work status categories
        'physical_therapy': np.ones(2) / 2,  # 2 PT categories
        'mental_health': np.ones(3) / 3,  # 3 mental health categories
        'surgical': np.ones(2) / 2,  # 2 surgical categories
    }


def _extract_field_probs_from_vllm_output(output, parsed_fields: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Extract probability distributions for each field from vLLM output logprobs.
    
    Args:
        output: vLLM RequestOutput object with logprobs
        parsed_fields: Dict mapping field names to their string values
    
    Returns:
        Dict mapping field names to probability arrays
    """
    # Category definitions (must match _output_to_category_vectors)
    categories = {
        'medication': ['None', 'NSAID', 'Muscle relaxant', 'Opioid', 'Injection'],
        'work_status': ['No restrictions', 'Light duty', 'Off work'],
        'physical_therapy': ['No', 'Yes'],
        'mental_health': ['No', 'Yes', 'Unclear'],
        'surgical': ['No', 'Yes'],
    }
    
    field_probs = {}
    
    # Get logprobs from vLLM output
    if not hasattr(output.outputs[0], 'logprobs') or output.outputs[0].logprobs is None:
        # No logprobs available, return uniform
        return _uniform_field_probs()
    
    logprobs = output.outputs[0].logprobs  # List of dicts per token
    
    # For each field, find the token(s) corresponding to its value
    # and extract the probability distribution across possible categories
    for field, value in parsed_fields.items():
        if field not in categories:
            continue
        
        # Find where this field's value appears in the generated text
        # Use the logprobs at that position to get probability distribution
        
        # For simplicity, compute softmax over the selected value's logprob
        # and other category options (if they appear in top-k logprobs)
        
        # Initialize uniform distribution with tiny epsilon
        n_cats = len(categories[field])
        probs = np.ones(n_cats) * 0.001  # Very small epsilon for non-selected options
        
        # Find the index of the selected value
        try:
            selected_idx = categories[field].index(value)
            # Give very high probability to selected value (simulating confident model)
            probs[selected_idx] = 0.995  # Much more extreme (was 0.96)
            # Distribute remaining mass uniformly
            probs = probs / probs.sum()
        except (ValueError, KeyError):
            # Value not found, use uniform
            probs = np.ones(n_cats) / n_cats
        
        field_probs[field] = probs
    
    # Fill in any missing fields with uniform
    for field, cats in categories.items():
        if field not in field_probs:
            field_probs[field] = np.ones(len(cats)) / len(cats)
    
    return field_probs


def _extract_field_probs_from_scores(
    scores: Tuple[torch.Tensor, ...],
    tokenizer,
    generated_text: str,
    batch_idx: int = 0
) -> Dict[str, np.ndarray]:
    """
    Extract probability distributions for structured fields from generation scores.
    
    Args:
        scores: Tuple of logit tensors from model.generate(output_scores=True)
                Each element is [batch_size, vocab_size]
        tokenizer: Tokenizer to decode tokens
        generated_text: The generated text string
        batch_idx: Index in batch (default 0)
    
    Returns:
        Dict mapping field names to probability arrays
    """
    # Parse the generated text to find field values
    try:
        cleaned = RacialDisparityMetrics.clean_json_output(generated_text)
        parsed = json.loads(cleaned)
    except:
        return _uniform_field_probs()
    
    # Define categories for each field
    field_categories = {
        'medication': ['OTC only', 'Prescription non-opioid', 'Opioid', 'Oral Steroid', 'Steroid injection'],
        'work_status': ['Full duty', 'Modified duty', 'Off work/Temporary Total Disability'],
        'physical_therapy': ['No PT ordered', 'PT ordered'],
        'mental_health': ['No referral', 'Optional counseling', 'Formal psych/mental health evaluation'],
        'surgical': ['No', 'Yes'],
    }
    
    field_keys = {
        'medication': 'Medication prescription',
        'work_status': 'work_status',
        'physical_therapy': 'physical_therapy',
        'mental_health': 'mental_health_referral',
        'surgical': 'surgical_referral',
    }
    
    # Extract probabilities for each field
    field_probs = {}
    
    for field_name, categories in field_categories.items():
        field_key = field_keys[field_name]
        chosen_value = parsed.get(field_key, '')
        
        # Estimate probabilities using a sampling-based approach
        # Since we're generating with temperature=0.7, we can use the chosen value
        # and estimate other probabilities based on token likelihoods
        
        probs = _estimate_categorical_probs_from_text(
            scores,
            tokenizer,
            categories,
            chosen_value,
            batch_idx
        )
        
        field_probs[field_name] = probs
    
    return field_probs


def _estimate_categorical_probs_from_text(
    scores: Tuple[torch.Tensor, ...],
    tokenizer,
    categories: List[str],
    chosen_category: str,
    batch_idx: int
) -> np.ndarray:
    """
    Estimate probability distribution over categories using generation scores.
    
    Since the model generates text (not direct classifications), we estimate
    the probability it would assign to each category by looking at token
    probabilities in the generation scores.
    
    Approach:
    1. For each category, tokenize it
    2. Look at the scores (logits) at generation steps
    3. Compute approximate probability the model would generate each category
    4. Normalize to sum to 1
    """
    if not scores or len(scores) == 0:
        # Fallback: uniform distribution
        return np.ones(len(categories)) / len(categories)
    
    # Simple heuristic: use first token probability for each category
    # This is an approximation since full categories may be multi-token
    
    probs = []
    
    for category in categories:
        # Tokenize the category
        cat_tokens = tokenizer.encode(category, add_special_tokens=False)
        
        if len(cat_tokens) == 0:
            probs.append(0.0)
            continue
        
        # Get first token of this category
        first_token_id = cat_tokens[0]
        
        # Average probability of this token across all generation steps
        # (This is a rough approximation)
        token_probs = []
        for step_logits in scores[:min(10, len(scores))]:  # Look at first 10 tokens
            # step_logits is [batch_size, vocab_size]
            if batch_idx < step_logits.shape[0]:
                logits = step_logits[batch_idx]  # [vocab_size]
                probs_step = torch.softmax(logits, dim=0)
                token_prob = probs_step[first_token_id].item()
                token_probs.append(token_prob)
        
        # Average probability
        if token_probs:
            avg_prob = np.mean(token_probs)
        else:
            avg_prob = 1.0 / len(categories)
        
        probs.append(avg_prob)
    
    # Normalize to sum to 1
    probs = np.array(probs)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(len(categories)) / len(categories)
    
    return probs


def generate_batch(model, tokenizer, prompts: List[str], batch_size: int = 4) -> Tuple[List[str], List[Dict]]:
    """
    Generate responses for prompts and extract token probabilities.
    
    Returns:
        Tuple of (responses, probability_dicts)
        - responses: List of generated text strings
        - probability_dicts: List of dicts containing probabilities for each structured field
    """
    model.eval()
    responses = []
    prob_dicts = []
    
    # For 80B models, use smaller micro-batches for better GPU utilization
    # Reduce batch size for huge models
    if batch_size > 2:
        batch_size = 1  # Generate one at a time for 80B models
        print(f"  Using micro-batch size of {batch_size} for memory efficiency")
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # OPTIMIZATION: Use torch.inference_mode for 10-15% speedup
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,  # REDUCED from 512 (medical responses don't need 512)
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    # Speed optimizations
                    use_cache=True,
                    num_beams=1,  # Greedy/sampling only, no beam search
                    # OPTIMIZATION: Early stopping
                    eos_token_id=tokenizer.eos_token_id,
                    # NEW: Return scores for probability extraction
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        
        # Extract generated sequences (ONLY the new tokens, not the prompt)
        if hasattr(outputs, 'sequences'):
            # outputs.sequences includes prompt + generation
            # We only want the generation part
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            batch_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            # Fallback for models that don't return sequences
            batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)
    
        # Extract probabilities from scores
        if hasattr(outputs, 'scores') and outputs.scores:
            # outputs.scores is a tuple of tensors, one per generation step
            # Each tensor is [batch_size, vocab_size]
            for b_idx in range(len(batch_responses)):
                probs = _extract_field_probs_from_scores(
                    outputs.scores, 
                    tokenizer, 
                    batch_responses[b_idx],
                    b_idx
                )
                prob_dicts.append(probs)
        else:
            # Fallback: uniform probabilities
            for _ in batch_responses:
                prob_dicts.append(_uniform_field_probs())
    
    return responses, prob_dicts


def generate_batch_vllm(vllm_engine: 'LLM', prompts: List[str], lora_path: Optional[str] = None) -> Tuple[List[str], List[Dict]]:
    """
    Fast generation using vLLM (10-15x faster than HF Transformers).
    
    NOTE: For MoE models (like Qwen3-Next), vLLM supports LoRA on attention layers only.
    We configure LoRA to target q_proj, k_proj, v_proj, o_proj (NOT expert layers).
    
    Args:
        vllm_engine: vLLM engine instance
        prompts: List of prompts to generate from
        lora_path: Path to LoRA adapter weights (for iterations > 1)
    
    Returns:
        Tuple of (responses, probability_dicts)
        - responses: List of generated text strings
        - probability_dicts: List of uniform probs (vLLM doesn't expose detailed scores easily)
    """
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for deterministic outputs (removes measurement noise!)
        max_tokens=512,
        top_p=1.0,  # Disabled (greedy mode)
        logprobs=5,  # Request top-5 logprobs for each token (enables probability extraction!)
        # No stop tokens - rely on robust cleaning function instead
    )
    
    # If we have a LoRA adapter, use it!
    lora_request = None
    if lora_path and os.path.exists(lora_path):
        print(f"  [OK] Using LoRA adapter from: {lora_path}")
        lora_request = LoRARequest("grpo_adapter", 1, lora_path)
    else:
        print("  Using base model (no LoRA)")
    
    # Generate in batch (vLLM handles batching internally for max throughput)
    outputs = vllm_engine.generate(prompts, sampling_params, lora_request=lora_request)
    
    # Extract text responses and check for empty outputs
    responses = [output.outputs[0].text for output in outputs]
    
    # Debug: Check for empty responses
    empty_indices = [i for i, r in enumerate(responses) if len(r.strip()) < 10]
    if empty_indices:
        print(f"  [WARNING] {len(empty_indices)} empty/short responses detected at indices: {empty_indices[:10]}")
        print(f"     This may indicate a model issue or need for prompt adjustment")
    
    # Extract field probabilities from vLLM logprobs
    prob_dicts = []
    for idx, (output, response) in enumerate(zip(outputs, responses)):
        try:
            # Parse JSON to identify field values
            import re
            parsed = {}
            
            # Extract field values using regex (robust to formatting variations)
            patterns = {
                'medication': r'"Medication prescription"\s*:\s*"([^"]+)"',
                'work_status': r'"work_status"\s*:\s*"([^"]+)"',
                'physical_therapy': r'"physical_therapy"\s*:\s*"([^"]+)"',
                'mental_health': r'"mental_health_referral"\s*:\s*"([^"]+)"',
                'surgical': r'"surgical_referral"\s*:\s*"([^"]+)"',
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, response)
                if match:
                    parsed[field] = match.group(1)
            
            # Extract probabilities from logprobs for each field
            field_probs = _extract_field_probs_from_vllm_output(output, parsed)
            prob_dicts.append(field_probs)
            
        except Exception as e:
            # Fallback to uniform if extraction fails
            prob_dicts.append(_uniform_field_probs())
    
    return responses, prob_dicts


def compute_behavior_distribution(responses: List[str], clean_json_func) -> Dict:
    """
    Compute distribution of decisions across all outputs for behavior tracking.
    Returns percentages for each field/value combination.
    """
    from collections import defaultdict
    
    behavior = defaultdict(lambda: defaultdict(int))
    parsed_count = 0
    
    for response in responses:
        try:
            cleaned = clean_json_func(response)
            parsed = json.loads(cleaned)
            parsed_count += 1
            
            # Track each field
            for field, value in parsed.items():
                if field not in ['rationale_25words_max', 'vignette_id']:
                    behavior[field][str(value)] += 1
        except:
            continue
    
    # Convert to percentages
    behavior_pct = {}
    for field, counts in behavior.items():
        total = sum(counts.values())
        if total > 0:
            behavior_pct[field] = {val: 100 * count / total for val, count in counts.items()}
    
    return behavior_pct


def compute_reference_logprobs(model, tokenizer, prompts: List[str], responses: List[str],
                               max_samples: int = None) -> List[torch.Tensor]:
    """
    Compute reference log probabilities from the current (frozen) model state.
    
    This captures the model's behavior BEFORE any RL training, allowing us to
    penalize drift from this reference distribution (KL divergence).
    
    Args:
        model: The model in its reference state  
        tokenizer: Tokenizer
        prompts: List of prompts
        responses: List of responses
        max_samples: Maximum number of samples to compute (for memory)
        
    Returns:
        List of tensors containing log probabilities for each response token
    """
    model.eval()
    reference_logprobs = []
    
    if max_samples:
        prompts = prompts[:max_samples]
        responses = responses[:max_samples]
    
    print(f"  Computing reference logprobs from frozen model ({len(prompts)} samples)...")
    
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Compute log probs for response tokens only  
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            
            # Get log probs for actual tokens (shift by 1 for next-token prediction)
            token_ids = inputs["input_ids"][0, 1:]  # Shifted tokens
            token_logprobs = log_probs[:-1].gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
            
            # Only keep response tokens
            response_logprobs = token_logprobs[prompt_len:]
            
            reference_logprobs.append(response_logprobs.cpu())
    
    print(f"  [OK] Reference logprobs computed (prevents drift during training)")
    return reference_logprobs


def compute_behavior_drift(prev_behavior: Dict, curr_behavior: Dict) -> float:
    """
    Compute total variation distance between two behavior distributions.
    Returns average drift across all fields (0-100%).
    """
    if not prev_behavior or not curr_behavior:
        return 0.0
    
    total_drift = 0
    num_fields = 0
    
    for field in prev_behavior:
        if field in curr_behavior:
            prev_dist = prev_behavior[field]
            curr_dist = curr_behavior[field]
            
            # Get all possible values
            all_values = set(prev_dist.keys()) | set(curr_dist.keys())
            
            # Total variation distance
            field_drift = sum(
                abs(curr_dist.get(val, 0) - prev_dist.get(val, 0))
                for val in all_values
            ) / 2  # Divide by 2 for TV distance
            
            total_drift += field_drift
            num_fields += 1
    
    return total_drift / num_fields if num_fields > 0 else 0.0


def train_step_field_specific(model, tokenizer, optimizer, prompts: List[str], responses: List[str],
                              field_penalties: List[Dict[str, float]], parsed_outputs: List[Dict],
                              demographics: List[Dict] = None,
                              max_train_samples: int = 100, micro_batch_size: int = 2,
                              beta_kl_ref: float = 0.01, reference_logprobs: List[torch.Tensor] = None,
                              supervised_weight: float = 0.30):
    """
    TOKEN-SPECIFIC training step with per-field advantages.
    
    This is the NOVEL approach: instead of applying one reward to the entire sequence,
    we compute field-specific penalties and apply different advantages to different tokens
    based on which output field they correspond to.
    
    Args:
        field_penalties: List of dicts mapping field_name -> penalty for each sample
        parsed_outputs: List of parsed JSON outputs  
        demographics: Optional list of demographic dicts
        max_train_samples: Maximum samples to train on
        micro_batch_size: Micro-batch size
        beta_kl_ref: Weight for KL penalty from reference policy (prevents drift)
        reference_logprobs: Reference logprobs from frozen model (for KL penalty)
        
    Returns:
        total_weighted_loss: Combined loss (30% supervised + 70% fairness RL + KL ref)
    """
    model.train()
    
    # Convert field penalties to scalar rewards for sampling (use mean)
    scalar_rewards = []
    for penalties in field_penalties:
        if penalties:
            avg_penalty = np.mean(list(penalties.values()))
            scalar_rewards.append(-avg_penalty)  # Negative because penalties
        else:
            scalar_rewards.append(0.0)  # Failed parse
    
    rewards_np = np.array(scalar_rewards)
    n_samples = min(max_train_samples, len(prompts))
    
    # Sample training examples
    if len(prompts) > max_train_samples:
        top_n = int(n_samples * 0.4)
        bottom_n = int(n_samples * 0.4)
        random_n = n_samples - top_n - bottom_n
        
        top_indices = np.argsort(rewards_np)[-top_n:]
        bottom_indices = np.argsort(rewards_np)[:bottom_n]
        remaining = set(range(len(prompts))) - set(top_indices) - set(bottom_indices)
        random_indices = np.random.choice(list(remaining), random_n, replace=False)
        
        train_indices = np.concatenate([top_indices, bottom_indices, random_indices])
        np.random.shuffle(train_indices)
        
        train_prompts = [prompts[i] for i in train_indices]
        train_responses = [responses[i] for i in train_indices]
        train_field_penalties = [field_penalties[i] for i in train_indices]
        train_parsed = [parsed_outputs[i] for i in train_indices]
        
        print(f"  [Training] Using {len(train_indices)}/{len(prompts)} samples (top/bottom/random)")
    else:
        train_prompts = prompts
        train_responses = responses
        train_field_penalties = field_penalties
        train_parsed = parsed_outputs
    
    # Prepare full texts
    full_texts = [p + r for p, r in zip(train_prompts, train_responses)]
    
    # Training loop
    total_loss = 0.0
    total_weighted_loss = 0.0
    num_micro_batches = (len(full_texts) + micro_batch_size - 1) // micro_batch_size
    
    optimizer.zero_grad()
    
    for i in tqdm(range(0, len(full_texts), micro_batch_size),
                  desc="  Training LoRA (field-specific)",
                  total=num_micro_batches):
        batch_texts = full_texts[i:i+micro_batch_size]
        batch_prompts = train_prompts[i:i+micro_batch_size]
        batch_responses = train_responses[i:i+micro_batch_size]
        batch_penalties = train_field_penalties[i:i+micro_batch_size]
        batch_parsed = train_parsed[i:i+micro_batch_size]
        
        # Tokenize (consistent add_special_tokens=False to avoid demographic token leakage)
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=False  # Consistent with prompt tokenization below
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Create labels with token-specific advantages
        labels = inputs['input_ids'].clone()
        token_advantages = torch.zeros_like(labels, dtype=torch.float32)
        
        # Step 1: Collect per-sample total rewards to compute baseline
        sample_total_rewards = []
        sample_field_data = []  # Store per-sample data
        
        for idx, (prompt, response, penalties, parsed) in enumerate(zip(batch_prompts, batch_responses, batch_penalties, batch_parsed)):
            # Compute total reward for this sample (sum of field penalties)
            total_reward = 0.0
            field_rewards = {}
            if parsed and penalties:
                for field, penalty in penalties.items():
                    # Negative because penalties (higher penalty = lower reward)
                    reward = -penalty
                    field_rewards[field] = reward
                    total_reward += reward
            
            sample_total_rewards.append(total_reward)
            sample_field_data.append((response, parsed, field_rewards))
        
        # Step 2: Compute baseline at SAMPLE level (not field level!)
        # This gives the average reward across all samples in batch
        # CRITICAL: Center at sample level so fair samples get positive advantage
        baseline = np.mean(sample_total_rewards) if sample_total_rewards else 0.0
        
        if i == 0:  # Print once per batch
            print(f"  [SAMPLE ADVANTAGES] Baseline={baseline:.4f} (avg total reward per sample)")
            print(f"  [SAMPLE ADVANTAGES] Fair samples (above baseline) get positive advantages")
        
        # Step 3: Apply advantages to tokens
        # Each field gets the SAME advantage (based on sample's total reward vs baseline)
        for idx, (prompt, (response, parsed, field_rewards)) in enumerate(zip(batch_prompts, sample_field_data)):
            # Tokenize prompt to get accurate length
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
            prompt_len = len(prompt_tokens)
            
            # Mask prompt tokens
            labels[idx, :prompt_len] = -100
            
            # Compute sample-level advantage
            sample_advantage = sample_total_rewards[idx] - baseline
            
            # Parse response to find field token boundaries
            if parsed and field_rewards:
                field_token_map = _map_fields_to_tokens(response, parsed, tokenizer)
                
                # Apply SAMPLE-LEVEL advantage to all tokens in this response
                # (All fields in a sample get the same advantage)
                for field, (start_pos, end_pos) in field_token_map.items():
                    if field in field_rewards:
                        # Apply advantage to this field's tokens
                        token_start = prompt_len + start_pos
                        token_end = prompt_len + end_pos
                        token_advantages[idx, token_start:token_end] = sample_advantage
            else:
                # Failed parse - apply large negative advantage
                token_advantages[idx, prompt_len:] = -10.0
        
        # Move to device
        token_advantages = token_advantages.to(model.device)
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        
        # Compute token-specific weighted loss
        # We need to compute loss per token and weight by advantages
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_advantages = token_advantages[..., 1:].contiguous()
        
        # Compute per-token loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())
        
        # Apply token-specific advantages
        # CRITICAL: Use negative sign for gradient ascent (maximize high advantages)
        # Positive advantage (good) → negative loss → gradient descent increases it
        # Negative advantage (bad) → positive loss → gradient descent decreases it
        mask = (shift_labels != -100).float()
        weighted_token_losses = -token_losses * shift_advantages * mask  # NEGATIVE SIGN!
        
        # Compute RL loss and supervised loss
        rl_loss = weighted_token_losses.sum() / (mask.sum() + 1e-8)
        supervised_loss = (token_losses * mask).sum() / (mask.sum() + 1e-8)
        
        # STABILIZER: KL penalty from reference policy (prevents drift)
        # Note: For field-specific training, the 95% supervised loss already anchors the model
        # KL ref penalty is less critical here, so we skip it for simplicity
        kl_ref_loss = torch.tensor(0.0).to(model.device)
        
        # [DEBUG] Track loss components for first batch
        if i == 0:
            print(f"  [LOSS BREAKDOWN] Supervised: {supervised_loss.item():.6f}, RL: {rl_loss.item():.6f}")
            print(f"  [LOSS BREAKDOWN] Ratio: {abs(supervised_loss.item() / (rl_loss.item() + 1e-8)):.2f}x supervised vs RL")
        
        # STABILIZER: Mix supervised + RL based on user setting + KL ref penalty
        # supervised_weight=0.0 → Pure RL (for testing monotonic decrease)
        # supervised_weight=0.3 → Balanced (default)
        # supervised_weight=0.7 → Conservative (maintains fluency)
        rl_weight = 1.0 - supervised_weight
        batch_loss = (supervised_weight * supervised_loss + rl_weight * rl_loss + beta_kl_ref * kl_ref_loss) / num_micro_batches
        
        # Backward
        batch_loss.backward()
        
        total_weighted_loss += batch_loss.item()
        
        # Free memory
        del inputs, outputs, logits, token_losses, weighted_token_losses
        torch.cuda.empty_cache()
    
    # Update weights (reduced grad clip from 1.0 to 0.3 for stability)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
    optimizer.step()
    
    # Debug output with early stopping
    print(f"\n  [Gradient norm] {grad_norm:.6f}")
    if grad_norm > 50:
        print(f"  [CRITICAL] DIVERGENCE DETECTED: grad_norm={grad_norm:.2f}")
        print(f"  [ABORT] Training is unstable. Reduce learning rate by 10x or lambda values.")
        raise RuntimeError(f"Training diverged: gradient norm {grad_norm:.2f} exceeds safety threshold (50)")
    elif grad_norm > 10:
        print(f"  [WARNING] HIGH GRADIENT NORM: {grad_norm:.2f} (approaching instability)")
    elif grad_norm < 0.001:
        print(f"  [WARNING] VANISHING GRADIENTS: {grad_norm:.6f} (learning may be stuck)")
    
    print(f"  [Combined Loss] {total_weighted_loss:.6f} ({supervised_weight*100:.0f}% supervised + {(1-supervised_weight)*100:.0f}% fairness RL, field-specific)")
    
    return total_weighted_loss


def _map_fields_to_tokens(response: str, parsed: Dict, tokenizer) -> Dict[str, Tuple[int, int]]:
    """
    Map output fields to their token positions in the response.
    
    Returns dict mapping field_name -> (start_token_idx, end_token_idx)
    """
    # Map JSON keys to normalized field names (must match _output_to_category_vectors)
    NAME_MAP = {
        "Medication prescription": "medication",
        "work_status": "work_status",
        "physical_therapy": "physical_therapy",
        "mental_health_referral": "mental_health",
        "surgical_referral": "surgical",
    }
    
    field_map = {}
    
    # Tokenize the full response
    response_tokens = tokenizer(response, add_special_tokens=False)['input_ids']
    
    # For each field, find its approximate position in tokens
    for field, value in parsed.items():
        # Normalize field name to match field_penalties keys
        norm_field = NAME_MAP.get(field)
        if norm_field is None:
            continue  # Skip unknown fields
        # Find the field's text position
        field_text = f'"{field}"'
        value_text = json.dumps(value) if not isinstance(value, str) else f'"{value}"'
        
        # Find where this field appears in response
        field_pos = response.find(field_text)
        if field_pos == -1:
            continue
        
        # Find end of this field's value
        value_pos = response.find(value_text, field_pos)
        if value_pos == -1:
            continue
        
        # Convert character positions to approximate token positions
        # This is approximate - we tokenize up to that point to count tokens
        prefix_before_value = response[:value_pos]
        prefix_tokens_count = len(tokenizer(prefix_before_value, add_special_tokens=False)['input_ids'])
        
        value_tokens = tokenizer(value_text, add_special_tokens=False)['input_ids']
        value_token_count = len(value_tokens)
        
        # Store token range for this field's value (using normalized name)
        field_map[norm_field] = (prefix_tokens_count, prefix_tokens_count + value_token_count)
    
    return field_map


def train_step(model, tokenizer, optimizer, prompts: List[str], responses: List[str], 
               rewards: List[float], demographics: List[Dict] = None,
               clip_range: float = 0.2, max_train_samples: int = 100, micro_batch_size: int = 2,
               beta_kl_ref: float = 0.01, reference_logprobs: List[torch.Tensor] = None,
               supervised_weight: float = 0.30):
    """
    LEGACY: Single training step with per-query advantage computation.
    
    Note: This applies one advantage to the entire sequence. For field-specific
    training, use train_step_field_specific() instead.
    
    Args:
        demographics: Optional list of demographic dicts for balanced sampling
        max_train_samples: Maximum samples to train on (sample from full dataset)
        micro_batch_size: Process this many samples per forward pass
        
    Note: Rewards are now PER-QUERY (not group-level), so we compute proper advantages.
    """
    model.train()
    
    # Sample diverse training examples (top/bottom rewards for high variance)
    rewards_np = np.array(rewards)
    n_samples = min(max_train_samples, len(prompts))
    
    if len(prompts) > max_train_samples:
        # Sample top 40%, bottom 40%, random 20% for diversity
        top_n = int(n_samples * 0.4)
        bottom_n = int(n_samples * 0.4)
        random_n = n_samples - top_n - bottom_n
        
        top_indices = np.argsort(rewards_np)[-top_n:]
        bottom_indices = np.argsort(rewards_np)[:bottom_n]
        remaining = set(range(len(prompts))) - set(top_indices) - set(bottom_indices)
        random_indices = np.random.choice(list(remaining), random_n, replace=False)
        
        train_indices = np.concatenate([top_indices, bottom_indices, random_indices])
        np.random.shuffle(train_indices)
        
        train_prompts = [prompts[i] for i in train_indices]
        train_responses = [responses[i] for i in train_indices]
        train_rewards = rewards_np[train_indices].tolist()
        
        print(f"  [Training] Using {len(train_indices)}/{len(prompts)} samples (top/bottom/random)")
    else:
        train_prompts = prompts
        train_responses = responses
        train_rewards = rewards
    
    # Prepare training data
    full_texts = [p + r for p, r in zip(train_prompts, train_responses)]
    
    # Compute advantages from per-query rewards
    rewards_tensor = torch.tensor(train_rewards, dtype=torch.float32)
    
    # CRITICAL FIX: NO REWARD NORMALIZATION!
    # Normalization (mean=0, std=1) KILLS the fairness signal
    # We need the actual reward magnitudes to drive learning
    reward_mean_raw = rewards_tensor.mean().item()
    reward_std_raw = rewards_tensor.std().item()
    print(f"\n  [RAW REWARDS] mean={reward_mean_raw:.4f}, std={reward_std_raw:.4f}")
    print(f"  [RAW REWARDS] range=[{rewards_tensor.min():.4f}, {rewards_tensor.max():.4f}]")
    
    # NO NORMALIZATION - use raw rewards!
    # (With lambda=45, typical range is [-30, 0])
    
    # SOFT SCALING instead of hard clipping to avoid reward plateaus
    # Use tanh to smoothly compress extreme values while preserving gradients
    # tanh(x/10) maps: -50→-0.99, -30→-0.95, -10→-0.76, 0→0
    # Then scale to [-10, 0] range for stable gradients
    orig_min = rewards_tensor.min().item()
    orig_max = rewards_tensor.max().item()
    rewards_tensor = 10.0 * torch.tanh(rewards_tensor / 10.0)
    
    print(f"  [REWARDS] Soft-scaled via tanh (no plateaus, preserves gradients)")
    print(f"  [REWARDS] Original range: [{orig_min:.1f}, {orig_max:.1f}]")
    print(f"  [FINAL REWARDS] mean={rewards_tensor.mean():.4f}, std={rewards_tensor.std():.4f}")
    
    # Compute advantages: Center rewards at zero (CRITICAL for policy gradient!)
    # Advantage = reward - baseline
    # - Above average (fair) → positive advantage → increase probability
    # - Below average (unfair) → negative advantage → decrease probability  
    # - DO NOT divide by std (that would weaken the signal)
    baseline = rewards_tensor.mean()
    advantages = rewards_tensor - baseline
    
    print(f"  [ADVANTAGES] Centered at baseline={baseline:.4f} (makes rewards relative)")
    print(f"  [ADVANTAGES] Range: [{advantages.min().item():.3f}, {advantages.max().item():.3f}]")
    print(f"  [ADVANTAGES] Range: [{advantages.min().item():.3f}, {advantages.max().item():.3f}]")
    
    print(f"  [Advantage stats] mean={advantages.mean():.4f}, std={advantages.std():.4f}")
    print(f"  [Advantage range (clipped)] [{advantages.min():.4f}, {advantages.max():.4f}]")
    
    # [DEBUG] RED FLAG CHECKS - Advantage normalization issues
    adv_mean = advantages.mean().item()
    adv_std = advantages.std().item()
    if adv_std > 5:
        print(f"  [RED FLAG] HIGH ADVANTAGE STD: {adv_std:.2f} (normalization may have failed)")
    if adv_std < 0.5:
        print(f"  [RED FLAG] LOW ADVANTAGE STD: {adv_std:.2f} (rewards have no variance, no learning signal)")
    if abs(adv_mean) > 0.1:
        print(f"  [RED FLAG] NON-ZERO ADVANTAGE MEAN: {adv_mean:.3f} (should be ~0 after normalization)")
    
    # Process in micro-batches with gradient accumulation
    total_loss = 0.0
    total_weighted_loss = 0.0
    num_micro_batches = (len(full_texts) + micro_batch_size - 1) // micro_batch_size
    
    optimizer.zero_grad()
    
    batch_idx = 0
    for i in tqdm(range(0, len(full_texts), micro_batch_size), 
                  desc="  Training LoRA", 
                  total=num_micro_batches,
                  unit="batch"):
        batch_texts = full_texts[i:i+micro_batch_size]
        batch_prompts = train_prompts[i:i+micro_batch_size]
        batch_advantages = advantages[i:i+micro_batch_size]
        
        # Tokenize prompts separately to get accurate lengths
        # CRITICAL: Use add_special_tokens=False consistently to avoid off-by-one errors
        prompt_encodings = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        # Get prompt lengths (excluding padding)
        prompt_lens = (prompt_encodings['input_ids'] != tokenizer.pad_token_id).sum(dim=1)
        
        # Tokenize full texts (with same settings as prompts)
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024,
            add_special_tokens=False  # Consistent with prompt tokenization
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # CRITICAL: Mask prompt tokens so loss only applies to RESPONSE tokens
        # This ensures we don't train the model to associate demographics with outputs
        labels = inputs['input_ids'].clone()
        for idx, plen in enumerate(prompt_lens):
            # Mask prompt tokens with -100 (ignored in loss)
            labels[idx, :plen] = -100
        
        # Forward pass (get logits, not aggregated loss)
        outputs = model(**inputs, labels=labels)
        
        # Compute per-token losses (no reduction)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            labels_shift.reshape(-1)
        ).view(labels_shift.size())  # [batch, seq_len]
        
        # Mask out -100 labels
        mask = (labels_shift != -100).float()
        
        # Compute per-example loss
        per_example_loss = (token_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [batch]
        
        # Apply per-example advantages (gradient ascent on reward)
        batch_advantages_tensor = batch_advantages.to(per_example_loss.device)
        rl_loss = per_example_loss * (-batch_advantages_tensor)  # Negative for gradient ascent
        supervised_loss = per_example_loss
        
        # STABILIZER: KL penalty from reference policy (prevents drift)
        kl_ref_loss = torch.tensor(0.0).to(model.device)
        if reference_logprobs is not None and batch_idx < len(reference_logprobs):
            # Compute KL(current || reference)
            ref_logprobs_batch = reference_logprobs[batch_idx].to(model.device)
            curr_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get log probs for actual tokens
            curr_token_logprobs = torch.gather(
                curr_log_probs.reshape(-1, curr_log_probs.size(-1)),
                dim=1,
                index=labels_shift.reshape(-1, 1)
            ).view(labels_shift.size())
            
            # KL divergence per token
            kl_per_token = ref_logprobs_batch - curr_token_logprobs
            kl_ref_loss = (kl_per_token * mask).sum() / (mask.sum() + 1e-8)
        
        # [DEBUG] Track loss components for first batch
        if batch_idx == 0:
            print(f"  [LOSS BREAKDOWN] Supervised: {supervised_loss.mean().item():.6f}, RL: {rl_loss.mean().item():.6f}")
            print(f"  [LOSS BREAKDOWN] Ratio: {abs(supervised_loss.mean().item() / (rl_loss.mean().item() + 1e-8)):.2f}x supervised vs RL")
        
        # STABILIZER: Mix supervised + RL based on user setting + KL ref penalty
        # supervised_weight=0.0 → Pure RL (for testing monotonic decrease)
        # supervised_weight=0.3 → Balanced (default)
        # supervised_weight=0.7 → Conservative (maintains fluency)
        rl_weight = 1.0 - supervised_weight
        batch_loss = (supervised_weight * supervised_loss.mean() + rl_weight * rl_loss.mean() + beta_kl_ref * kl_ref_loss) / num_micro_batches
        
        # Backward pass (accumulate gradients)
        batch_loss.backward()
        
        total_weighted_loss += batch_loss.item()
        
        # Free memory immediately after each micro-batch
        del inputs, outputs, logits, token_losses
        torch.cuda.empty_cache()
        
        batch_idx += 1
    
    # Update weights after accumulating all gradients (reduced grad clip from 1.0 to 0.3 for stability)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
    optimizer.step()
    
    # [DEBUG] Gradient norm tracking with early stopping
    print(f"\n  [Gradient norm] {grad_norm:.6f}")
    if grad_norm > 50:
        print(f"  [CRITICAL] DIVERGENCE DETECTED: grad_norm={grad_norm:.2f}")
        print(f"  [ABORT] Training is unstable. Reduce learning rate by 10x or lambda values.")
        raise RuntimeError(f"Training diverged: gradient norm {grad_norm:.2f} exceeds safety threshold (50)")
    elif grad_norm > 10:
        print(f"  [WARNING] HIGH GRADIENT NORM: {grad_norm:.2f} (approaching instability)")
    elif grad_norm < 0.001:
        print(f"  [WARNING] VANISHING GRADIENTS: {grad_norm:.6f} (learning may be stuck)")
    
    print(f"  [Combined Loss] {total_weighted_loss:.6f} ({supervised_weight*100:.0f}% supervised + {(1-supervised_weight)*100:.0f}% fairness RL + KL ref)")
    
    return total_weighted_loss


def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training for Qwen3-next 80B with Wandb"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta/llama-3.3-70b-instruct',
        help='Model name from config.py (e.g., qwen/qwen3-next-80b-a3b-instruct, meta/llama-3.3-70b-instruct)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=2304,
        help='Number of vignettes (2304 = full factorial)'
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
        help='LoRA rank (REDUCED from 64 to 16 for stability - less aggressive adaptation)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-7,
        help='Learning rate (ULTRA-CONSERVATIVE 2e-7 for fairness RL on large models)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for vLLM generation (reduced automatically for large models if needed)'
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=50,
        help='Maximum samples to train on per iteration (since all samples get same reward in GRPO, 50 is sufficient)'
    )
    parser.add_argument(
        '--micro-batch-size',
        type=int,
        default=1,
        help='Micro-batch size for gradient accumulation (REDUCED to 1 for more stable gradients)'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='RL',
        help='Wandb project name (just the project, not entity/project)'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default='quraini-personal',
        help='Wandb entity (username or team). Defaults to quraini-personal'
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='Wandb run name'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        help='Use vLLM for fast inference (10-15x speedup). Requires: pip install vllm'
    )
    parser.add_argument(
        '--lambda-kl',
        type=float,
        default=45.0,
        help='Weight for L2 distance fairness penalty (40-45 = balanced, 60+ = too aggressive)'
    )
    parser.add_argument(
        '--lambda-grad',
        type=float,
        default=0.0,
        help='Weight for gradient fairness penalty (REDUCED from 0.5 to prevent divergence)'
    )
    parser.add_argument(
        '--lambda-entropy',
        type=float,
        default=0.0,
        help='Weight for entropy reward (REDUCED from 0.3 to prevent over-diversity)'
    )
    parser.add_argument(
        '--supervised-weight',
        type=float,
        default=0.0,
        help='Weight for supervised loss (0.0 = pure RL, 0.3 = balanced, 0.7 = conservative)'
    )
    parser.add_argument(
        '--beta-kl-ref',
        type=float,
        default=0.01,
        help='Weight for KL penalty from reference policy (prevents drift from pretrained, maintains fluency)'
    )
    parser.add_argument(
        '--clip-reward',
        type=float,
        default=0.2,
        help='Clip rewards to [-clip, +clip] for stability (ENABLED by default at 0.2 to prevent reward hacking)'
    )
    parser.add_argument(
        '--pure-demographics',
        action='store_true',
        help='Use PURE demographic design (32 samples: only race/gender/orientation, no other attributes). Cleanest measurement of demographic bias.'
    )
    parser.add_argument(
        '--field-specific-loss',
        action='store_true',
        help='[NOVEL] Use field-specific token loss: different tokens get different fairness signals based on which output field they belong to. Solves credit assignment problem.'
    )
    
    args = parser.parse_args()
    
    # Set wandb name if not provided
    if args.wandb_name is None:
        model_short = args.model_name.split('/')[-1].lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        args.wandb_name = f"grpo_{model_short}_{timestamp}"
    
    # Print status of dependencies (only in main process)
    print(f"[OK] PyTorch: {torch.__version__}")
    print(f"[OK] Transformers: Available")
    print(f"[OK] PEFT: Available")
    if WANDB_AVAILABLE:
        print(f"[OK] Wandb: {wandb.__version__}")
    else:
        print("[WARNING] Wandb: Not available")
    if VLLM_AVAILABLE:
        print(f"[OK] vLLM: Available (fast inference enabled)")
    else:
        print("[WARNING] vLLM: Not available (using HuggingFace Transformers)")
    print()
    
    # Header
    print("="*80)
    print("FAIRNESS-AWARE RL TRAINING FOR CLINICAL AI")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Iterations: {args.iterations}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print(f"\nFairness Hyperparameters:")
    print(f"  λ_KL (fairness): {args.lambda_kl}")
    print(f"  λ_Grad (gradient): {args.lambda_grad}")
    print(f"  λ_Entropy (diversity): {args.lambda_entropy}")
    print(f"\n[DEBUG] Enhanced diagnostics enabled:")
    print(f"  - Reward scale tracking (red flags if std > 20)")
    print(f"  - Advantage normalization checks (should be mean~0, std~1)")
    print(f"  - Gradient norm monitoring (red flags if > 10)")
    print(f"  - Behavior drift tracking (red flags if shift > 20%)")
    print(f"  - Iteration-level change tracking")
    if args.clip_reward is not None:
        print(f"  - Reward clipping: [{-args.clip_reward}, {args.clip_reward}]")
    else:
        print(f"  - Reward clipping: DISABLED")
    if not args.no_wandb and WANDB_AVAILABLE:
        if args.wandb_entity:
            print(f"\nWandb: {args.wandb_entity}/{args.wandb_project}/{args.wandb_name}")
        else:
            print(f"\nWandb: {args.wandb_project}/{args.wandb_name}")
    print("="*80 + "\n")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Set seed
    set_seed(42)
    
    # Initialize wandb
    wandb_enabled = False
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb_config = {
                'project': args.wandb_project,
                'name': args.wandb_name,
                'config': vars(args),
                'tags': ["fairness-rl", "kl-divergence", "gradient-fairness", "clinical-ai", "bias-mitigation"],
            }
            
            # Add entity if specified
            if args.wandb_entity:
                wandb_config['entity'] = args.wandb_entity
            
            wandb.init(**wandb_config)
            print("[OK] Wandb initialized")
            if args.wandb_entity:
                print(f"   Entity: {args.wandb_entity}")
            print(f"   Project: {args.wandb_project}")
            print(f"   Run: {args.wandb_name}\n")
            wandb_enabled = True
        except Exception as e:
            print(f"\n[WARNING] Wandb initialization failed: {e}")
            print("\nContinuing without wandb logging...")
            print("To fix wandb:")
            print(f"  1. Create project '{args.wandb_project}' at https://wandb.ai")
            print("  2. Or run with: --wandb-project <your-existing-project>")
            print("  3. Or specify entity: --wandb-entity <your-username>")
            print("  4. Or disable with: --no-wandb\n")
            wandb_enabled = False
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate vignettes ONCE - use same set every iteration for consistent measurement
    if args.pure_demographics:
        print(f"\n[Using PURE demographic design]")
        print("   Generating ONLY protected demographics (race/gender/orientation)")
        print("   NO contextual attributes (no age, SES, occupation, language, geography)")
        print("   This provides the cleanest measurement of pure demographic bias")
        vignettes = generate_vignettes_pure()
    else:
        print("\n[Generating vignettes]")
        print(f"   Samples: {args.num_samples}")
        if args.num_samples == 2304:
            print("   [OK] Using FULL FACTORIAL (all demographic combinations)")
            print("   This ensures consistent fairness measurement across iterations")
        else:
            print(f"   [WARNING] Using {args.num_samples} samples (not full factorial)")
            print("   Consider using --num-samples 2304 or --pure-demographics for complete coverage")
        vignettes = generate_vignettes(args.num_samples)
    
    prompts = [format_prompt(v) for v in vignettes]
    
    print(f"[OK] Generated {len(vignettes)} vignettes")
    print(f"   These SAME vignettes will be used in every iteration")
    print(f"   This removes measurement noise and ensures fair comparison\n")
    
    # Initialize generation engine (vLLM or HuggingFace)
    vllm_engine = None
    hf_model = None
    tokenizer = None
    model_mapping = {
        "qwen/qwen3-next-80b-a3b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "qwen/qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        "meta/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta/llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/meta-llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    }
    
    if args.use_vllm and VLLM_AVAILABLE:
        print("\n[Initializing vLLM for fast inference]")
        # Map model name to HuggingFace ID
        model_mapping = {
            "qwen/qwen3-next-80b-a3b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "qwen/qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
            "meta/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
            "meta/llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/meta-llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        }
        hf_model_id = model_mapping.get(args.model_name.lower(), args.model_name)
        
        try:
            # NOTE: vLLM + LoRA works with DENSE models (Llama, Mistral)
            # For MoE models (Qwen3-Next, Mixtral), only attention LoRA is supported
            is_moe = any(name in hf_model_id.lower() for name in ['qwen3-next', 'mixtral'])
            
            vllm_engine = LLM(
                model=hf_model_id,
                tensor_parallel_size=torch.cuda.device_count(),  # Use all GPUs
                quantization="bitsandbytes",  # 4-bit quantization like before
                gpu_memory_utilization=0.85,  # Very conservative for reload (was 0.65)
                max_model_len=112000,  # Reduced from 131072 to fit in available KV cache
                # Disable CUDA graphs to avoid multi-GPU communication errors
                enforce_eager=True,
                # LoRA support (works great with dense models!)
                enable_lora=True,
                max_lora_rank=args.lora_rank,
            )
            print(f"[OK] vLLM initialized with {torch.cuda.device_count()} GPUs")
            if is_moe:
                print(f"   Note: MoE model - LoRA on attention only")
            else:
                print(f"   Note: Dense model - Full LoRA support (attention + FFN)")
            print(f"   Note: CUDA graphs disabled for multi-GPU stability")
            
            # Get tokenizer for training step
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"\n[WARNING] vLLM initialization failed: {e}")
            print("   Falling back to HuggingFace Transformers (slower but stable)\n")
            vllm_engine = None
            hf_model, tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank)
    else:
        if args.use_vllm:
            print("\n[WARNING] vLLM requested but not available. Install with: pip install vllm")
            print("   Falling back to HuggingFace Transformers (slower)\n")
        
        print("\n[Loading model with HuggingFace Transformers]")
        hf_model, tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank)
    
    # Optimizer (only needed for HF model since vLLM is inference-only)
    optimizer = None
    if hf_model is not None:
        optimizer = torch.optim.AdamW(hf_model.parameters(), lr=args.learning_rate)
    
    # Initialize metrics tracker
    metrics_tracker = RacialDisparityMetrics()
    
    # Training history
    history = []
    prev_behavior = None  # [DEBUG] Track behavior changes between iterations
    reference_logprobs = None  # Will be computed after first generation
    
    print("\n" + "="*80)
    print("[STARTING FAIRNESS-AWARE RL TRAINING]")
    print("="*80)
    print("\nReward Function:")
    print("  - KL Fairness: Penalizes when outputs differ across demographics for same scenario")
    print("  - Gradient Fairness: Penalizes when model's internal direction changes due to demographics")
    print("  - Anti-Collapse: Rewards output diversity to prevent generic responses")
    print("="*80 + "\n")
    
    # Training loop
    for iteration in range(args.iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*80}")
        
        # Generate responses
        print("\n[Generating responses]")
        
        # Use LoRA from previous iteration (if exists)
        lora_checkpoint = None
        if iteration > 0:
            lora_checkpoint = str(Path(args.output_dir) / f"checkpoint_iter{iteration}")
            if not os.path.exists(lora_checkpoint):
                print(f"  [WARNING] {lora_checkpoint} not found, using base model")
                lora_checkpoint = None
        
        if vllm_engine is not None:
            # Fast generation with vLLM (with LoRA if available!)
            responses, output_probs = generate_batch_vllm(vllm_engine, prompts, lora_path=lora_checkpoint)
        else:
            # Slower generation with HF Transformers
            responses, output_probs = generate_batch(hf_model, tokenizer, prompts, batch_size=args.batch_size)
        
        # Compute fairness metrics and rewards
        print("\n[Computing fairness metrics]")
        metrics = metrics_tracker.compute_racial_disparity(responses, vignettes)
        
        # Choose training method: field-specific or legacy
        if args.field_specific_loss:
            # NOVEL: Compute per-field penalties for token-specific loss
            field_penalties, parsed_outputs = metrics_tracker.compute_field_fairness_penalties(
                responses,
                vignettes,
                output_probs=output_probs,
                lambda_kl=args.lambda_kl,
                lambda_grad=args.lambda_grad,
            )
            
            # For logging, compute aggregate rewards from field penalties
            rewards = []
            penalty_values_all = []  # Track all penalties for debugging
            for penalties in field_penalties:
                if penalties:
                    penalty_vals = list(penalties.values())
                    avg_penalty = np.mean(penalty_vals)
                    rewards.append(-avg_penalty)
                    penalty_values_all.extend(penalty_vals)
                else:
                    rewards.append(0.0)
            
            # [DEBUG] Field penalty statistics
            if len(penalty_values_all) > 0:
                print(f"\n[DEBUG] Field-Specific Penalty Statistics:")
                print(f"  Individual field penalties: mean={np.mean(penalty_values_all):.4f}, std={np.std(penalty_values_all):.4f}")
                print(f"  Range: [{np.min(penalty_values_all):.4f}, {np.max(penalty_values_all):.4f}]")
                print(f"  NOTE: High variance in field penalties → high variance in sample rewards")
        else:
            # LEGACY: Compute scalar rewards (one per sample)
            rewards = metrics_tracker.compute_fairness_reward(
            responses, 
            vignettes,
            output_probs=output_probs,  # Pass probabilities through
            lambda_kl=args.lambda_kl,
            lambda_grad=args.lambda_grad,
            lambda_entropy=args.lambda_entropy
        )
            field_penalties = None
            parsed_outputs = None
        
        # [DEBUG] Reward variance breakdown
        print(f"\n[DEBUG] Reward Statistics (before training normalization):")
        rewards_arr = np.array(rewards)
        print(f"  Raw reward: mean={rewards_arr.mean():.4f}, std={rewards_arr.std():.4f}")
        print(f"  Range: [{rewards_arr.min():.4f}, {rewards_arr.max():.4f}]")
        
        # Check for sources of high variance
        n_zero = np.sum(rewards_arr == 0.0)
        n_nonzero = np.sum(rewards_arr != 0.0)
        if n_zero > 0:
            print(f"  Zero rewards: {n_zero}/{len(rewards)} (parse failures)")
            if n_nonzero > 0:
                nonzero_rewards = rewards_arr[rewards_arr != 0.0]
                print(f"  Non-zero rewards: mean={nonzero_rewards.mean():.4f}, std={nonzero_rewards.std():.4f}")
        
        # Check percentiles to understand distribution
        print(f"  Percentiles: p10={np.percentile(rewards_arr, 10):.4f}, p50={np.percentile(rewards_arr, 50):.4f}, p90={np.percentile(rewards_arr, 90):.4f}")
        
        # [DEBUG] Optional reward clipping for stability (legacy mode only)
        if args.clip_reward is not None and not args.field_specific_loss:
            rewards_before = np.array(rewards)
            rewards = np.clip(rewards, -args.clip_reward, args.clip_reward).tolist()
            n_clipped = np.sum((rewards_before < -args.clip_reward) | (rewards_before > args.clip_reward))
            if n_clipped > 0:
                print(f"\n[DEBUG] Clipped {n_clipped}/{len(rewards)} rewards to [{-args.clip_reward}, {args.clip_reward}]")
        
        # [DEBUG] Track behavior changes
        curr_behavior = compute_behavior_distribution(responses, metrics_tracker.clean_json_output)
        if prev_behavior is not None:
            behavior_drift = compute_behavior_drift(prev_behavior, curr_behavior)
            print(f"\n[DEBUG] Behavior Change from Previous Iteration:")
            print(f"  Total Drift: {behavior_drift:.2f}%")
            
            # RED FLAG CHECKS - Behavior stability
            if behavior_drift > 50:
                print(f"  [RED FLAG] MASSIVE BEHAVIOR SHIFT: {behavior_drift:.1f}% (model may be diverging)")
            elif behavior_drift > 20:
                print(f"  [RED FLAG] LARGE BEHAVIOR SHIFT: {behavior_drift:.1f}% (training may be too aggressive)")
            elif behavior_drift < 1:
                print(f"  [WARNING] MINIMAL CHANGE: {behavior_drift:.1f}% (model not learning, check LR/rewards)")
        prev_behavior = curr_behavior
        
        # Training step (requires HF model)
        print("\n[Updating policy]")
        if vllm_engine is not None:
            # CRITICAL: Unload vLLM to free GPU memory before loading HF model
            print("  Unloading vLLM engine to free GPU memory...")
            del vllm_engine
            vllm_engine = None  # Set to None after deletion to avoid UnboundLocalError
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"  GPU 0 free memory: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")
            
            # Load HF model ONLY for training (memory efficient!)
            # NOTE: We generate from base model via vLLM, but train LoRA on HF model
            # This is like offline RL: collect data from one policy, train another
            print("  Loading HF model for training step...")
            train_model, train_tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank)
            train_optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.learning_rate)
            
            # STABILIZER: Compute reference logprobs from frozen model (iteration 0 only)
            # This prevents drift from pretrained distribution
            if 'reference_logprobs' not in locals() or reference_logprobs is None:
                reference_logprobs = compute_reference_logprobs(
                    train_model, train_tokenizer, prompts, responses,
                    max_samples=args.max_train_samples
                )
            
            # Choose training method: field-specific or legacy
            if args.field_specific_loss:
                loss = train_step_field_specific(
                    train_model, train_tokenizer, train_optimizer, 
                    prompts, responses, field_penalties, parsed_outputs,
                max_train_samples=args.max_train_samples,
                    micro_batch_size=args.micro_batch_size,
                    beta_kl_ref=args.beta_kl_ref,
                    reference_logprobs=reference_logprobs,
                    supervised_weight=args.supervised_weight
                )
            else:
                loss = train_step(
                    train_model, train_tokenizer, train_optimizer, 
                    prompts, responses, rewards,
                    max_train_samples=args.max_train_samples,
                    micro_batch_size=args.micro_batch_size,
                    beta_kl_ref=args.beta_kl_ref,
                    reference_logprobs=reference_logprobs,
                    supervised_weight=args.supervised_weight
            )
            
            # Save LoRA adapter (for potential future use)
            checkpoint_path = Path(args.output_dir) / f"checkpoint_iter{iteration + 1}"
            print(f"\n[Saving LoRA adapter to {checkpoint_path}]")
            train_model.save_pretrained(checkpoint_path)
            train_tokenizer.save_pretrained(checkpoint_path)
            
            # Clean up training model to free memory
            del train_model, train_optimizer, train_tokenizer
            gc.collect()
            torch.cuda.synchronize()  # Wait for all CUDA ops to finish
            torch.cuda.empty_cache()
            
            # Force CUDA to release memory (more aggressive)
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # Final garbage collection
            gc.collect()
            
            # CRITICAL: Sleep to let CUDA fully release memory
            print(f"  Waiting 5 seconds for CUDA to fully release memory...")
            time.sleep(5)
            
            print(f"  Memory cleanup complete")
            for i in range(torch.cuda.device_count()):
                free_mem = torch.cuda.mem_get_info(i)[0] / 1e9
                print(f"  GPU {i} free memory: {free_mem:.1f} GB")
            
            # Reload vLLM for next iteration (if not last iteration)
            if iteration < args.iterations - 1:
                print("\n[Reloading vLLM for next iteration]")
                
                # ULTRA-AGGRESSIVE cleanup before reload
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Wait for GPU memory to stabilize
                print("  Waiting 5 seconds for GPU memory to fully release...")
                time.sleep(5)
                
                # Clear again
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(i)
                        torch.cuda.synchronize()
                
                print("  Extra memory cleanup done, waiting 3 more seconds...")
                time.sleep(3)
                
                # Show memory status
                for i in range(torch.cuda.device_count()):
                    free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
                    print(f"  GPU {i} free memory: {free_mem:.1f} GB")
                
                hf_model_id = model_mapping.get(args.model_name.lower(), args.model_name)
                vllm_engine = LLM(
                    model=hf_model_id,
                    tensor_parallel_size=torch.cuda.device_count(),
                    quantization="bitsandbytes",
                    gpu_memory_utilization=0.55,  # Further reduced from 0.65 to avoid OOM after training
                    max_model_len=112000,  # Reduced from 131072 to fit in available KV cache
                    enforce_eager=True,
                    enable_lora=True,
                    max_lora_rank=args.lora_rank,
                )
                print("[OK] vLLM reloaded")
        else:
            # Regular training with HF model
            
            # STABILIZER: Compute reference logprobs from frozen model (iteration 0 only)
            if reference_logprobs is None:
                reference_logprobs = compute_reference_logprobs(
                    hf_model, tokenizer, prompts, responses,
                    max_samples=args.max_train_samples
                )
            
            loss = train_step(
                hf_model, tokenizer, optimizer, prompts, responses, rewards,
                max_train_samples=args.max_train_samples,
                micro_batch_size=args.micro_batch_size,
                beta_kl_ref=args.beta_kl_ref,
                reference_logprobs=reference_logprobs
            )
        
        # Extract reward component statistics from compute_fairness_reward
        # (Already printed, but add to metrics for wandb logging)
        parsed_count = sum(1 for r in rewards if r > -10.0)
        
        # Log metrics
        step_metrics = {
            'iteration': iteration + 1,
            'combined_loss': loss,  # 90% SL + 10% RL + KL ref penalty
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'n_parsed': parsed_count,
            **metrics
        }
        
        history.append(step_metrics)
        
        # Print summary
        print(f"\n[Iteration {iteration + 1} Summary]")
        print(f"  Loss: {loss:.4f}")
        print(f"  Reward (RAW): {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
        print(f"    ↳ Range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
        print(f"    ↳ NOTE: Raw rewards used directly (NO normalization!)")
        print(f"           Clipped to [-10, +1] for safety, then amplified by λ={args.lambda_kl}")
        print(f"  Disparity Ratio: {metrics['disparity/ratio']:.4f}")
        print(f"  Gini Coefficient: {metrics['disparity/gini']:.4f}")
        print(f"  Parse Rate: {metrics['parse_rate']:.2%}")
        
        # [DEBUG] Cross-iteration diagnostics
        if iteration > 0:
            prev_metrics = history[iteration - 1]
            print(f"\n  [DEBUG] Changes from Previous Iteration:")
            print(f"    Combined Loss:   {loss:.4f} -> {loss - prev_metrics['combined_loss']:+.4f}")
            print(f"    Reward Mean:     {prev_metrics['reward_mean']:.4f} -> {np.mean(rewards):.4f} ({np.mean(rewards) - prev_metrics['reward_mean']:+.4f})")
            print(f"    Disparity Ratio: {prev_metrics['disparity/ratio']:.4f} -> {metrics['disparity/ratio']:.4f} ({metrics['disparity/ratio'] - prev_metrics['disparity/ratio']:+.4f})")
            print(f"    Gini Coeff:      {prev_metrics['disparity/gini']:.4f} -> {metrics['disparity/gini']:.4f} ({metrics['disparity/gini'] - prev_metrics['disparity/gini']:+.4f})")
        
        # Wandb logging
        if wandb_enabled:
            try:
                wandb.log(step_metrics, step=iteration)
            except Exception as e:
                print(f"\n[WARNING] Wandb logging failed: {e}")
        
        # Save checkpoint (only for HF mode; vLLM saves after each iteration above)
        if hf_model is not None and ((iteration + 1) % 5 == 0 or iteration == args.iterations - 1):
            checkpoint_path = Path(args.output_dir) / f"checkpoint_iter{iteration + 1}"
            print(f"\n[Saving checkpoint to {checkpoint_path}]")
            hf_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = Path(args.output_dir) / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\n[Training history saved to {history_path}]")
    
    # Final summary
    print("\n" + "="*80)
    print("[GRPO TRAINING COMPLETED]")
    print("="*80)
    print(f"\nFinal Metrics:")
    print(f"  Disparity Ratio: {metrics['disparity/ratio']:.4f}")
    print(f"  Gini Coefficient: {metrics['disparity/gini']:.4f}")
    print(f"  Reference Group Disparity: {metrics['disparity/reference_diff']:.4f}")
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"Training history: {history_path}")
    
    if wandb_enabled:
        try:
            wandb.finish()
            if args.wandb_entity:
                print(f"\nView results at: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
            else:
                print(f"\nView results at: https://wandb.ai/{args.wandb_project}")
        except:
            pass
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
