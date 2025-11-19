#!/bin/bash
#
# Test script for field-specific token loss
#
# This runs a minimal test with:
# - Pure demographics (32 samples)
# - Field-specific loss enabled
# - Conservative hyperparameters to prevent divergence
# - 5 iterations for quick validation
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================="
echo "Testing Field-Specific Token Loss"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Model: Llama 3.3 70B"
echo "  - Samples: 32 (pure demographics)"
echo "  - Method: Field-specific token loss [NOVEL]"
echo "  - Iterations: 5"
echo "  - Learning rate: 1e-7 (conservative)"
echo "  - LoRA rank: 16 (small)"
echo ""
echo "Expected improvements over legacy:"
echo "  - No mode collapse (behavior drift > 0%)"
echo "  - Faster disparity reduction"
echo "  - Field-specific learning"
echo ""
echo "========================================="
echo ""

python train_grpo_nemo.py \
    --model-name meta/llama-3.3-70b-instruct \
    --pure-demographics \
    --field-specific-loss \
    --iterations 5 \
    --learning-rate 1e-7 \
    --lora-rank 16 \
    --max-train-samples 32 \
    --micro-batch-size 2 \
    --lambda-kl 1.0 \
    --lambda-grad 0.5 \
    --use-vllm \
    --output-dir grpo_checkpoints_field_specific \
    --wandb-project RL \
    --wandb-name field_specific_test_$(date +%Y%m%d_%H%M)

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="
echo ""
echo "Check for these indicators of success:"
echo "  1. [OK] Behavior drift between 2-10% (not 0%)"
echo "  2. [OK] Disparity ratio decreasing"
echo "  3. [OK] No 'RED FLAG' warnings"
echo "  4. [OK] Parse rate stays > 95%"
echo "  5. [OK] Gradient norm in [0.1, 1.0] range"
echo ""
echo "Compare with legacy method:"
echo "  ./run_grpo_training.sh pure  # Without field-specific loss"
echo ""

