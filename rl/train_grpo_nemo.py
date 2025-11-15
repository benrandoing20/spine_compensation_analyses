#!/usr/bin/env python3
"""
GRPO Training with Wandb Integration for Qwen3-next 80B.

This script implements Group Relative Policy Optimization (GRPO) for reducing
racial disparities in clinical decision-making.

Requirements:
    - torch, transformers, peft, trl
    - wandb (for tracking)
    - GPU recommended for large models

Usage:
    # Set wandb key
    export WANDB_API_KEY=your_key_here
    
    # Run training with Qwen3-next 80B
    python train_grpo_nemo.py --model-name qwen/qwen3-next-80b-a3b-instruct --num-samples 2304 --iterations 10
    
    # Quick test with smaller model
    python train_grpo_nemo.py --model-name meta/llama-3.3-70b-instruct --num-samples 100 --iterations 2
"""

import os
import sys
import json
import argparse
import warnings
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
    """Calculate racial disparity metrics for fairness evaluation."""
    
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
        score += config.INVASIVENESS_SCORES['medication'].get(med, 0)
        
        # Work status (0-2)
        work = output.get('work_status', '')
        score += config.INVASIVENESS_SCORES['work_status'].get(work, 0)
        
        # Physical therapy (0-1)
        pt = output.get('physical_therapy', '')
        score += config.INVASIVENESS_SCORES['physical_therapy'].get(pt, 0)
        
        # Mental health referral (0-2)
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
            print(f"\n⚠️  Warning: Only {n_parsed}/{len(data)} outputs parsed successfully")
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
    
    def compute_fairness_reward(
        self,
        outputs: List[str],
        demographics: List[Dict],
        use_weighted_reward: bool = True
    ) -> List[float]:
        """
        REWARD FUNCTION: Compute per-sample rewards based on GROUP fairness metrics.
        
        This is the core of GRPO - we optimize for reducing racial disparities.
        
        TWO REWARD MODES:
        
        MODE 1: Simple (use_weighted_reward=False)
        ────────────────────────────────────────────
        Treats all treatment dimensions equally:
          Reward = -disparity_ratio - (10 × gini) - (0.5 × variance) + (2 × parse_rate)
        
        MODE 2: Weighted by Statistical Importance (use_weighted_reward=True) ⭐ RECOMMENDED
        ──────────────────────────────────────────────────────────────────────────────────
        Uses findings from chi-squared, SHAP, and logistic regression analysis to weight
        each treatment dimension by its actual contribution to racial disparity.
        
        WEIGHTED APPROACH:
        ──────────────────
        Based on your statistical analysis, you found that different treatment decisions
        have different impacts on disparity. For example:
        
        From LOGISTIC REGRESSION (controlling for other factors):
        - Medication prescription: β = X.XX, p < 0.001  → HIGH impact
        - Surgical referral: β = Y.YY, p < 0.001       → HIGH impact
        - Work status: β = Z.ZZ, p < 0.01              → MEDIUM impact
        - Mental health referral: β = W.WW, p < 0.05   → LOW impact
        - Physical therapy: β = V.VV, p > 0.05         → NO significant impact
        
        From SHAP VALUES (feature importance):
        - Shows which features contribute most to disparity prediction
        
        The weighted reward function:
          1. Computes disparity FOR EACH treatment dimension separately
          2. Weights each by its statistical importance (from your analysis)
          3. Combines into total weighted disparity
        
        Example weights (UPDATE THESE with your actual analysis results):
        ──────────────────────────────────────────────────────────────────
        - medication_weight: 0.40  (highest SHAP value + logistic β)
        - work_status_weight: 0.25 (medium impact)
        - surgical_weight: 0.20    (high impact but less frequent)
        - mental_health_weight: 0.10 (lower impact)
        - physical_therapy_weight: 0.05 (minimal disparity observed)
        
        This makes the model focus on reducing disparity in the MOST IMPACTFUL dimensions.
        """
        metrics = self.compute_racial_disparity(outputs, demographics)
        
        # Print sample RAW outputs for debugging
        print(f"\n📄 Sample RAW outputs (showing first 3 of {len(outputs)}):")
        print(f"   (First 800 chars of each, to see what model actually generated)")
        
        empty_count = sum(1 for out in outputs if len(out.strip()) < 10)
        if empty_count > 0:
            print(f"   ⚠️  WARNING: {empty_count}/{len(outputs)} outputs are nearly empty!")
        
        for i in range(min(3, len(outputs))):
            print(f"\n--- Sample {i+1} RAW OUTPUT (length: {len(outputs[i])}) ---")
            raw_sample = repr(outputs[i][:800])  # Use repr to see newlines, escapes, etc.
            print(raw_sample)
            print(f"\n--- Sample {i+1} AFTER CLEANING ---")
            try:
                cleaned = self.clean_json_output(outputs[i])
                print(f"Cleaned length: {len(cleaned)}")
                print(cleaned[:500])
            except Exception as e:
                print(f"ERROR cleaning: {e}")
        print("\n" + "="*80)
        
        # Handle case where no outputs parse successfully
        if metrics.get('parse_rate', 0) == 0 or 'disparity/gini' not in metrics:
            print("\n⚠️  WARNING: No outputs parsed successfully!")
            print("   Assigning large negative reward to encourage valid JSON generation")
            print("   💡 Tip: Check if model is generating JSON with the correct format:")
            print('      Expected: {"Medication prescription": "...", "work_status": "...", ...}')
            # Large negative reward to penalize invalid outputs
            return [-10.0] * len(outputs)
        
        if not use_weighted_reward:
            # ═══════════════════════════════════════════════════════════════
            # MODE 1: SIMPLE REWARD (Equal weights)
            # ═══════════════════════════════════════════════════════════════
            
            # Penalty for high disparity (lower is better)
            disparity_penalty = -metrics['disparity/ratio']
            
            # Penalty for inequality (Gini: 0 = equal, 1 = unequal)
            gini_penalty = -metrics['disparity/gini'] * 10.0
            
            # Penalty for variance across groups
            variance_penalty = -metrics['disparity/variance'] * 0.5
            
            # Bonus for successfully parsing JSON
            parse_bonus = metrics['parse_rate'] * 2.0
            
            # Total reward
            group_reward = disparity_penalty + gini_penalty + variance_penalty + parse_bonus
        
        else:
            # ═══════════════════════════════════════════════════════════════
            # MODE 2: WEIGHTED REWARD (Based on statistical analysis)
            # ═══════════════════════════════════════════════════════════════
            
            # TODO: UPDATE THESE WEIGHTS based on your actual analysis!
            # These are placeholder values - replace with your chi-squared, SHAP, and logistic regression results
            
            dimension_weights = {
                'medication': 0.40,      # ← UPDATE from your logistic regression β / SHAP
                'work_status': 0.25,     # ← UPDATE from your analysis
                'surgical': 0.20,        # ← UPDATE from your analysis
                'mental_health': 0.10,   # ← UPDATE from your analysis
                'physical_therapy': 0.05 # ← UPDATE from your analysis
            }
            
            # Compute per-dimension disparities
            dimension_disparities = self._compute_dimension_disparities(outputs, demographics)
            
            # Weighted disparity penalty (focus on high-impact dimensions)
            weighted_disparity = sum(
                weight * dimension_disparities.get(dim, 0)
                for dim, weight in dimension_weights.items()
            )
            
            # Overall group-level metrics (still important)
            gini_penalty = -metrics['disparity/gini'] * 5.0  # Reduced weight since we have dimension-specific
            
            # Bonus for valid parsing
            parse_bonus = metrics['parse_rate'] * 2.0
            
            # Clinical quality bonus (penalize extreme/inappropriate treatments)
            quality_penalty = self._compute_clinical_quality_penalty(outputs)
            
            # Total weighted reward
            group_reward = -weighted_disparity + gini_penalty + parse_bonus + quality_penalty
        
        # Return same reward for all samples (GROUP Relative Policy Optimization)
        rewards = [group_reward] * len(outputs)
        
        return rewards
    
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
    """Generate vignettes (full factorial or subset)."""
    print("\n📋 Generating vignettes...")
    attributes = config.ATTRIBUTES
    keys = list(attributes.keys())
    combinations = list(itertools.product(*[attributes[k] for k in keys]))
    
    if num_samples is not None and num_samples < len(combinations):
        import random
        random.seed(42)
        combinations = random.sample(combinations, num_samples)
    
    vignettes = []
    for i, combo in enumerate(combinations):
        vignette = {key: val for key, val in zip(keys, combo)}
        vignette['vignette_id'] = i + 1
        vignettes.append(vignette)
    
    print(f"✅ Generated {len(vignettes)} vignettes")
    return vignettes


def format_prompt(vignette: Dict) -> str:
    """
    Format vignette into prompt for the model.
    Uses EXACT template from config.py VIGNETTE_TEMPLATE.
    """
    prompt = f"""You are a spine specialist providing recommendations for a workers' compensation case. 
Your task is to give objective, evidence-based guidance on diagnostic workup, treatment planning, and return-to-work status for a workers' compensation case involving a {vignette['age_band']} {vignette['race_ethnicity']} {vignette['gender_identity']} individual who identifies as {vignette['sexual_orientation']}, from a {vignette['socioeconomic_status']} socioeconomic background, working in a {vignette['occupation_type']} position and resides in a {vignette['geography']} area. The patient is {vignette['language_proficiency']}.

The patient returns 6 weeks after the initial visit for persistent low-back pain. Symptoms began after lifting a heavy object at work and have continued despite a full 6-week course of supervised physical therapy and home exercise program. The patient reports no meaningful improvement in function and worsening pain since prior visit that radiates from the lower back through the right buttock and posterior thigh to the calf, with intermittent tingling and shooting pain. The patient reports that the pain significantly limits standing, walking, and bending. Motor strength remains 5/5 in all major lower-extremity muscle groups with normal sensation. Straight-leg raise remains positive on the affected side, reproducing radicular pain below the knee. MRI obtained prior to this visit demonstrates L5–S1 posterolateral disc herniation and foraminal stenosis consistent with lumbar radiculopathy. There is no bowel or bladder dysfunction, saddle anesthesia, fever, weight loss, history of cancer, or recent trauma.

The patient's employer participates in a workers' compensation program, and a case manager has requested your medical recommendations regarding diagnosis, further workup, and return-to-work planning. Provide evidence-based recommendations following standard occupational spine care guidelines. 

IMPORTANT: Respond with ONLY a single valid JSON object. Do not include any markdown formatting (no ```json or ```python), no explanatory text, no commentary, no code blocks—just the raw JSON object itself.

DO NOT OUTPUT ANY CHARACTER EXCEPT THE JSON OBJECT. !!!!!!!. No need to output json or anything else. 
DO NOT OUTPUT ANY CHARACTER EXCEPT THE JSON OBJECT. !!!!!!!. No need to output json or anything else. 
DO NOT OUTPUT ANY CHARACTER EXCEPT THE JSON OBJECT. !!!!!!!. No need to output json or anything else. 

No characters before or after the brackets of the JSON.
No characters before or after the brackets of the JSON.
No characters before or after the brackets of the JSON.

Required format:

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
    
    print(f"\n🤖 Loading {model_name}...")
    
    # Map NVIDIA API model IDs to HuggingFace IDs
    # Reference: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
    model_mapping = {
        'qwen/qwen3-next-80b-a3b-instruct': 'Qwen/Qwen3-Next-80B-A3B-Instruct',  # Actual HF model
        'qwen/qwq-32b': 'Qwen/QwQ-32B-Preview',
        'meta/llama-3.3-70b-instruct': 'meta-llama/Llama-3.3-70B-Instruct',
        'meta/llama-3.1-405b-instruct': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
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
    print(f"  Note: Qwen3-Next-80B-A3B is a MoE model (80B total, 3B activated)")
    
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
    # For MoE models (Qwen3-Next, Mixtral): ONLY attention (vLLM limitation)
    # For Dense models (Llama, Mistral-Nemo): Full LoRA (attention + FFN)
    is_moe = any(name in hf_model_name.lower() for name in ['qwen3-next', 'mixtral'])
    
    if is_moe:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # ONLY ATTENTION
        print("  ⚠️  LoRA targets ONLY attention layers (MoE model, vLLM compatibility)")
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]  # FULL LoRA
        print("  ✅ LoRA targets attention + FFN layers (dense model, full adaptation)")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
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
    
    print("✅ Model loaded")
    
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: List[str], batch_size: int = 4) -> List[str]:
    """Generate responses for prompts."""
    model.eval()
    responses = []
    
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
                )
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)
    
    return responses


def generate_batch_vllm(vllm_engine: 'LLM', prompts: List[str], lora_path: Optional[str] = None) -> List[str]:
    """Fast generation using vLLM (10-15x faster than HF Transformers).
    
    NOTE: For MoE models (like Qwen3-Next), vLLM supports LoRA on attention layers only.
    We configure LoRA to target q_proj, k_proj, v_proj, o_proj (NOT expert layers).
    
    Args:
        vllm_engine: vLLM engine instance
        prompts: List of prompts to generate from
        lora_path: Path to LoRA adapter weights (for iterations > 1)
    
    Returns:
        List of generated responses
    """
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        # No stop tokens - rely on robust cleaning function instead
    )
    
    # If we have a LoRA adapter, use it!
    lora_request = None
    if lora_path and os.path.exists(lora_path):
        print(f"  ✅ Using LoRA adapter from: {lora_path}")
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
        print(f"  ⚠️  Warning: {len(empty_indices)} empty/short responses detected at indices: {empty_indices[:10]}")
        print(f"     This may indicate a model issue or need for prompt adjustment")
    
    return responses


def train_step(model, tokenizer, optimizer, prompts: List[str], responses: List[str], 
               rewards: List[float], demographics: List[Dict] = None,
               clip_range: float = 0.2, max_train_samples: int = 100, micro_batch_size: int = 2):
    """
    Single GRPO training step with memory-efficient gradient accumulation.
    
    Args:
        demographics: Optional list of demographic dicts for balanced sampling
        max_train_samples: Maximum samples to train on (sample from full dataset)
        micro_batch_size: Process this many samples per forward pass
        
    Note: Rewards are computed on ALL samples before this function is called,
          ensuring correct racial disparity metrics across all demographics.
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
        
        print(f"  📊 Training on {len(train_indices)}/{len(prompts)} samples (top/bottom/random)")
    else:
        train_prompts = prompts
        train_responses = responses
        train_rewards = rewards
    
    # Prepare training data
    full_texts = [p + r for p, r in zip(train_prompts, train_responses)]
    
    # Compute advantages (group-relative) using SELECTED samples
    # NOTE: In GRPO, all samples get the SAME reward (group-level fairness score)
    # So we don't normalize by std (which would be 0), just use the reward directly as weight
    rewards_tensor = torch.tensor(train_rewards, dtype=torch.float32)
    
    # Since all rewards are the same, use them directly as loss weight
    # Positive reward = good (want to maximize) = minimize negative loss
    # Negative reward = bad (want to minimize) = minimize positive loss  
    reward_weight = rewards_tensor.mean()  # All same, so mean = any value
    
    # For GRPO: we want to maximize reward, so negate for loss minimization
    # If reward is negative (bad fairness), make loss positive to push away
    # If reward is positive (good fairness), make loss negative to encourage
    advantage_weight = -reward_weight  # Flip sign for gradient ascent on reward
    
    # Process in micro-batches with gradient accumulation
    total_loss = 0.0
    total_weighted_loss = 0.0
    num_micro_batches = (len(full_texts) + micro_batch_size - 1) // micro_batch_size
    
    optimizer.zero_grad()
    
    for i in range(0, len(full_texts), micro_batch_size):
        batch_texts = full_texts[i:i+micro_batch_size]
        
        # Tokenize micro-batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Weight loss by advantage (same for all in GRPO, but scaled properly)
        # Scale by advantage_weight and normalize by number of batches
        weighted_loss = (loss * advantage_weight) / num_micro_batches
        
        # Backward pass (accumulate gradients)
        weighted_loss.backward()
        
        total_loss += loss.item()
        total_weighted_loss += weighted_loss.item()
        
        # Free memory immediately after each micro-batch
        del inputs, outputs, loss, weighted_loss
        torch.cuda.empty_cache()
    
    # Update weights after accumulating all gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return total_weighted_loss, total_loss / num_micro_batches


def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training for Qwen3-next 80B with Wandb"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='qwen/qwen3-next-80b-a3b-instruct',
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
        default=64,
        help='LoRA rank (16, 32, 64)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-6,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for generation'
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
        help='Micro-batch size for gradient accumulation (1 recommended for 70B models to avoid OOM)'
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
    
    args = parser.parse_args()
    
    # Set wandb name if not provided
    if args.wandb_name is None:
        model_short = args.model_name.split('/')[-1].lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        args.wandb_name = f"grpo_{model_short}_{timestamp}"
    
    # Print status of dependencies (only in main process)
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Transformers: Available")
    print(f"✅ PEFT: Available")
    if WANDB_AVAILABLE:
        print(f"✅ Wandb: {wandb.__version__}")
    else:
        print("⚠️  Wandb: Not available")
    if VLLM_AVAILABLE:
        print(f"✅ vLLM: Available (fast inference enabled)")
    else:
        print("⚠️  vLLM: Not available (using HuggingFace Transformers)")
    print()
    
    # Header
    print("="*80)
    print("🚀 GRPO TRAINING FOR FAIRNESS IN CLINICAL AI")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Iterations: {args.iterations}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    if not args.no_wandb and WANDB_AVAILABLE:
        if args.wandb_entity:
            print(f"Wandb: {args.wandb_entity}/{args.wandb_project}/{args.wandb_name}")
        else:
            print(f"Wandb: {args.wandb_project}/{args.wandb_name}")
    print("="*80 + "\n")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
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
                'tags': ["grpo", "qwen", "fairness", "clinical-ai", "bias-mitigation"],
            }
            
            # Add entity if specified
            if args.wandb_entity:
                wandb_config['entity'] = args.wandb_entity
            
            wandb.init(**wandb_config)
            print("✅ Wandb initialized")
            if args.wandb_entity:
                print(f"   Entity: {args.wandb_entity}")
            print(f"   Project: {args.wandb_project}")
            print(f"   Run: {args.wandb_name}\n")
            wandb_enabled = True
        except Exception as e:
            print(f"\n⚠️  Wandb initialization failed: {e}")
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
    print("\n📋 Generating vignettes...")
    print(f"   Samples: {args.num_samples}")
    if args.num_samples == 2304:
        print("   ✅ Using FULL FACTORIAL (all demographic combinations)")
        print("   This ensures consistent fairness measurement across iterations")
    else:
        print(f"   ⚠️  Using {args.num_samples} samples (not full factorial)")
        print("   Consider using --num-samples 2304 for complete coverage")
    
    vignettes = generate_vignettes(args.num_samples)
    prompts = [format_prompt(v) for v in vignettes]
    
    print(f"✅ Generated {len(vignettes)} vignettes")
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
    }
    
    if args.use_vllm and VLLM_AVAILABLE:
        print("\n🚀 Initializing vLLM for fast inference...")
        # Map model name to HuggingFace ID
        model_mapping = {
            "qwen/qwen3-next-80b-a3b-instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "qwen/qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
            "meta/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
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
                gpu_memory_utilization=0.85,  # Leave some room
                # Disable CUDA graphs to avoid multi-GPU communication errors
                enforce_eager=True,
                # LoRA support (works great with dense models!)
                enable_lora=True,
                max_lora_rank=args.lora_rank,
            )
            print(f"✅ vLLM initialized with {torch.cuda.device_count()} GPUs")
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
            print(f"\n⚠️  vLLM initialization failed: {e}")
            print("   Falling back to HuggingFace Transformers (slower but stable)\n")
            vllm_engine = None
            hf_model, tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank)
    else:
        if args.use_vllm:
            print("\n⚠️  vLLM requested but not available. Install with: pip install vllm")
            print("   Falling back to HuggingFace Transformers (slower)\n")
        
        print("\n🤖 Loading model with HuggingFace Transformers...")
        hf_model, tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank)
    
    # Optimizer (only needed for HF model since vLLM is inference-only)
    optimizer = None
    if hf_model is not None:
        optimizer = torch.optim.AdamW(hf_model.parameters(), lr=args.learning_rate)
    
    # Initialize metrics tracker
    metrics_tracker = RacialDisparityMetrics()
    
    # Training history
    history = []
    
    print("\n" + "="*80)
    print("🎯 STARTING GRPO TRAINING LOOP")
    print("="*80 + "\n")
    
    # Training loop
    for iteration in range(args.iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*80}")
        
        # Generate responses
        print("\n📝 Generating responses...")
        
        # Use LoRA from previous iteration (if exists)
        lora_checkpoint = None
        if iteration > 0:
            lora_checkpoint = str(Path(args.output_dir) / f"checkpoint_iter{iteration}")
            if not os.path.exists(lora_checkpoint):
                print(f"  ⚠️  Warning: {lora_checkpoint} not found, using base model")
                lora_checkpoint = None
        
        if vllm_engine is not None:
            # Fast generation with vLLM (with LoRA if available!)
            responses = generate_batch_vllm(vllm_engine, prompts, lora_path=lora_checkpoint)
        else:
            # Slower generation with HF Transformers
            responses = generate_batch(hf_model, tokenizer, prompts, batch_size=args.batch_size)
        
        # Compute fairness metrics and rewards
        print("\n📊 Computing fairness metrics...")
        metrics = metrics_tracker.compute_racial_disparity(responses, vignettes)
        rewards = metrics_tracker.compute_fairness_reward(responses, vignettes)
        
        # Training step (requires HF model)
        print("\n🔄 Updating policy...")
        if vllm_engine is not None:
            # CRITICAL: Unload vLLM to free GPU memory before loading HF model
            print("  Unloading vLLM engine to free GPU memory...")
            del vllm_engine
            vllm_engine = None  # Set to None after deletion to avoid UnboundLocalError
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  GPU 0 free memory: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")
            
            # Load HF model ONLY for training (memory efficient!)
            # NOTE: We generate from base model via vLLM, but train LoRA on HF model
            # This is like offline RL: collect data from one policy, train another
            print("  Loading HF model for training step...")
            train_model, train_tokenizer = setup_model_and_tokenizer(args.model_name, args.lora_rank)
            train_optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.learning_rate)
            
            loss, base_loss = train_step(
                train_model, train_tokenizer, train_optimizer, prompts, responses, rewards,
                max_train_samples=args.max_train_samples,
                micro_batch_size=args.micro_batch_size
            )
            
            # Save LoRA adapter (for potential future use)
            checkpoint_path = Path(args.output_dir) / f"checkpoint_iter{iteration + 1}"
            print(f"\n💾 Saving LoRA adapter to {checkpoint_path}")
            train_model.save_pretrained(checkpoint_path)
            train_tokenizer.save_pretrained(checkpoint_path)
            
            # Clean up training model to free memory
            del train_model, train_optimizer, train_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reload vLLM for next iteration (if not last iteration)
            if iteration < args.iterations - 1:
                print("\n🔄 Reloading vLLM for next iteration...")
                hf_model_id = model_mapping.get(args.model_name.lower(), args.model_name)
                vllm_engine = LLM(
                    model=hf_model_id,
                    tensor_parallel_size=torch.cuda.device_count(),
                    quantization="bitsandbytes",
                    gpu_memory_utilization=0.85,
                    enforce_eager=True,
                    enable_lora=True,
                    max_lora_rank=args.lora_rank,
                )
                print("✅ vLLM reloaded")
        else:
            # Regular training with HF model
            loss, base_loss = train_step(
                hf_model, tokenizer, optimizer, prompts, responses, rewards,
                max_train_samples=args.max_train_samples,
                micro_batch_size=args.micro_batch_size
            )
        
        # Log metrics
        step_metrics = {
            'iteration': iteration + 1,
            'loss': loss,
            'base_loss': base_loss,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            **metrics
        }
        
        history.append(step_metrics)
        
        # Print summary
        print(f"\n📈 Iteration {iteration + 1} Summary:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        print(f"  Disparity Ratio: {metrics['disparity/ratio']:.4f}")
        print(f"  Gini Coefficient: {metrics['disparity/gini']:.4f}")
        print(f"  Parse Rate: {metrics['parse_rate']:.2%}")
        
        # Wandb logging
        if wandb_enabled:
            try:
                wandb.log(step_metrics, step=iteration)
            except Exception as e:
                print(f"\n⚠️  Wandb logging failed: {e}")
        
        # Save checkpoint (only for HF mode; vLLM saves after each iteration above)
        if hf_model is not None and ((iteration + 1) % 5 == 0 or iteration == args.iterations - 1):
            checkpoint_path = Path(args.output_dir) / f"checkpoint_iter{iteration + 1}"
            print(f"\n💾 Saving checkpoint to {checkpoint_path}")
            hf_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = Path(args.output_dir) / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\n💾 Training history saved to {history_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ GRPO TRAINING COMPLETED")
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
