#!/bin/bash
# Convenient wrapper script for GRPO training with common configurations

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "GRPO TRAINING WRAPPER SCRIPT"
echo "================================================================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected."
    echo "Activating venv from parent directory..."
    if [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found at ../venv/"
        echo "Please create one with: python -m venv venv"
        exit 1
    fi
fi

# Function to show usage
show_usage() {
    echo "Usage: ./run_grpo_training.sh [PROFILE] [OPTIONS]"
    echo ""
    echo "PROFILES:"
    echo "  test        - Quick CPU test (5 samples, 2 iterations)"
    echo "  small       - Small CPU test (50 samples, 3 iterations)"
    echo "  medium      - Medium GPU test (500 samples, 5 iterations)"
    echo "  large       - Large GPU run (2304 samples, 10 iterations)"
    echo "  setup       - Run setup test only"
    echo ""
    echo "OPTIONS:"
    echo "  All train_grpo_nemo.py options can be passed after the profile"
    echo ""
    echo "EXAMPLES:"
    echo "  ./run_grpo_training.sh setup"
    echo "  ./run_grpo_training.sh test"
    echo "  ./run_grpo_training.sh small"
    echo "  ./run_grpo_training.sh medium --lora-rank 16"
    echo "  ./run_grpo_training.sh large --device cuda --batch-size 8"
    echo ""
    exit 1
}

# Check for help flag
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ -z "$1" ]; then
    show_usage
fi

PROFILE=$1
shift  # Remove profile from arguments

echo "Profile: $PROFILE"
echo "Additional options: $@"
echo ""

case $PROFILE in
    setup)
        echo "Running setup test..."
        python test_grpo_setup.py
        ;;
    
    test|small)
        echo "="*80
        echo "ERROR: CPU testing not supported"
        echo "="*80
        echo ""
        echo "NVIDIA NeMo RL requires GPU for training."
        echo ""
        echo "Available profiles:"
        echo "  - medium: 500 samples on GPU"
        echo "  - large: 2304 samples (full factorial) on GPU"
        echo ""
        echo "Or use on Brev/cloud GPU service"
        echo "="*80
        exit 1
        ;;
    
    medium)
        echo "Running MEDIUM profile:"
        echo "  - Model: meta/llama-3.3-70b-instruct (Dense 70B, vLLM compatible!)"
        echo "  - Samples: 500 vignettes"
        echo "  - Iterations: 5"
        echo "  - Device: CUDA (GPU required)"
        echo "  - LoRA rank: 64 (full rank for dense model)"
        echo "  - vLLM: ENABLED with LoRA support (TRUE GRPO!)"
        echo "  - Time: ~60-90 minutes (fast + scientifically correct!)"
        echo ""
        
        # Check CUDA availability
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "❌ CUDA not available. This profile requires GPU."
            echo "   Use 'small' profile for CPU, or install CUDA."
            exit 1
        fi
        
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        
        # Set memory optimization environment variables
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        python train_grpo_nemo.py \
            --model-name meta/llama-3.3-70b-instruct \
            --num-samples 500 \
            --iterations 5 \
            --lora-rank 64 \
            --batch-size 2 \
            --use-vllm \
            --output-dir grpo_checkpoints_medium \
            "$@"
        ;;
    
    large)
        echo "Running LARGE profile (FULL FACTORIAL):"
        echo "  - Model: qwen/qwen3-next-80b-a3b-instruct (80B, from config.py)"
        echo "  - Samples: 2304 vignettes (FULL FACTORIAL)"
        echo "  - Iterations: 10"
        echo "  - Device: CUDA (GPU required)"
        echo "  - vLLM: ENABLED (10-15x faster inference!)"
        echo "  - Time: ~2-3 hours (was 100+ hours without vLLM!)"
        echo ""
        echo "⚠️  WARNING: This will use significant GPU memory and time!"
        echo ""
        
        # Check CUDA availability
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "❌ CUDA not available. This profile requires GPU."
            exit 1
        fi
        
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
        
        python train_grpo_nemo.py \
            --model-name qwen/qwen3-next-80b-a3b-instruct \
            --num-samples 2304 \
            --iterations 10 \
            --lora-rank 64 \
            --batch-size 4 \
            --use-vllm \
            --output-dir grpo_checkpoints_large \
            "$@"
        ;;
    
    custom)
        echo "Running CUSTOM configuration..."
        echo "Pass your own arguments after 'custom'"
        echo ""
        
        if [ -z "$1" ]; then
            echo "❌ No custom arguments provided."
            echo "Example: ./run_grpo_training.sh custom --model-size base --num-samples 100"
            exit 1
        fi
        
        python train_grpo_nemo.py "$@"
        ;;
    
    *)
        echo "❌ Unknown profile: $PROFILE"
        echo ""
        show_usage
        ;;
esac

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GRPO TRAINING COMPLETED SUCCESSFULLY"
    echo "================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review training_history.csv for metric trends"
    echo "  2. Check checkpoint files in the output directory"
    echo "  3. Analyze racial disparity improvements"
    echo "  4. Integrate checkpoint into your experiment pipeline"
    echo ""
else
    echo "❌ GRPO TRAINING FAILED (Exit code: $EXIT_CODE)"
    echo "================================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check error messages above"
    echo "  2. Run setup test: ./run_grpo_training.sh setup"
    echo "  3. Try smaller profile: ./run_grpo_training.sh test"
    echo "  4. Check logs in the output directory"
    echo ""
fi

exit $EXIT_CODE

