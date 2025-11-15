# GRPO Training with NeMo RL Framework

Training LLMs with **Group Relative Policy Optimization (GRPO)** using NVIDIA NeMo RL framework to reduce racial disparities in clinical decision-making.

## 🚀 Quick Start Guide

### Prerequisites Checklist:
- [ ] **GPU with CUDA** (required for training)
- [ ] **Wandb account** → Get API key at https://wandb.ai/settings
- [ ] **HuggingFace account** → Get token at https://huggingface.co/settings/tokens
- [ ] **Llama access** (if using Llama models) → Accept license at https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

### Setup (5 minutes):

```bash
# 1. Create .env file with your keys
cd spine_compensation_analyses
cp env.template .env
nano .env  # Add your WANDB_API_KEY and HF_TOKEN

# 2. Authenticate with HuggingFace
source venv/bin/activate
huggingface-cli login  # Paste your HF_TOKEN

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training!
cd rl
./run_grpo_training.sh medium
```

## Overview

**What is GRPO?**
- Group Relative Policy Optimization is a reinforcement learning algorithm designed for fairness
- Optimizes model behavior to reduce disparities across demographic groups
- Unlike standard RL, GRPO explicitly penalizes policies that create unequal outcomes

**What does this do?**
1. Runs full factorial experiment (2304 vignettes with different demographics)
2. Calculates racial disparity metrics (Gini coefficient, disparity ratio, variance)
3. Uses GRPO + LoRA (Low-Rank Adaptation) for efficient fine-tuning
4. Produces checkpoints with reduced racial bias in treatment recommendations

## Requirements

⚠️ **NVIDIA GPU with CUDA is REQUIRED** ⚠️

NeMo Aligner is designed for GPU-based reinforcement learning training. CPU training is not supported.

### 1. Environment Setup

Create a `.env` file in the `spine_compensation_analyses` directory:

```bash
# Navigate to project root
cd spine_compensation_analyses

# Copy the template
cp env.template .env

# Edit with your keys
nano .env  # or use your favorite editor
```

Add your API keys to `.env`:

```bash
# Environment Variables for GRPO Training
WANDB_API_KEY=your_actual_wandb_key_here
HF_TOKEN=your_actual_hf_token_here
```

**Where to get these keys:**

1. **Wandb API Key:**
   - Go to https://wandb.ai/settings
   - Click "API keys" → "Create new API key"
   - Copy and paste into `.env`

2. **HuggingFace Token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token" → "Read" access
   - Copy and paste into `.env`
   
3. **For Llama Models (Gated Access):**
   - Go to https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
   - Click "Agree and access repository"
   - Wait for approval (usually instant)
   - Then authenticate with your HF token:
   ```bash
   source venv/bin/activate
   huggingface-cli login
   # Paste your HF_TOKEN from .env when prompted
   ```

### 2. Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate

# Install PyTorch with CUDA
# For CUDA 11.8:
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install NeMo and other dependencies
pip install -r requirements.txt
pip install nemo-toolkit[nlp]
pip install nemo-aligner
```

### 3. Verify Setup

```bash
python test_grpo_setup.py
```

## Quick Start (GPU Only)

**Medium training (30-60 minutes, GPU):**
```bash
./run_grpo_training.sh medium
```

**Full factorial (2-4 hours, GPU):**
```bash
./run_grpo_training.sh large
```

## Command-Line Options

```bash
python train_grpo_nemo.py [OPTIONS]

Options:
  --model-size {small,base,large}   Model size preset (default: small)
  --model-name TEXT                 Specific HuggingFace model name
  --num-samples INT                 Number of vignettes (default: 50)
  --iterations INT                  Number of GRPO iterations (default: 3)
  --output-dir TEXT                 Checkpoint directory (default: grpo_checkpoints)
  --lora-rank INT                   LoRA rank (default: 8)
  --learning-rate FLOAT             Learning rate (default: 1e-5)
  --batch-size INT                  Batch size (default: 4)
  --device {cpu,cuda}               Device (default: auto-detect)
  --use-nemo                        Use NeMo RL framework if available
```

### Example Commands

**CPU Testing:**
```bash
python train_grpo_nemo.py \
  --model-size small \
  --num-samples 20 \
  --iterations 2 \
  --device cpu
```

**GPU Production:**
```bash
python train_grpo_nemo.py \
  --model-name meta-llama/Llama-2-7b-hf \
  --num-samples 2304 \
  --iterations 10 \
  --lora-rank 16 \
  --batch-size 8 \
  --device cuda
```

**Using Custom Models:**
```bash
python train_grpo_nemo.py \
  --model-name meta/llama-3.3-70b-instruct \
  --num-samples 1000 \
  --iterations 5
```

## Understanding Metrics

### 1. Disparity Ratio (target: 1.0)
- Ratio of max/min mean invasiveness across racial groups
- Example: 1.2 = highest group gets 20% more invasive treatments
- Lower is better (1.0 = perfect equality)

### 2. Gini Coefficient (target: 0.0)
- Measures inequality (0 = perfect equality, 1 = perfect inequality)
- Borrowed from economics (wealth inequality)
- Applied to invasiveness across racial groups

### 3. Variance (target: 0.0)
- Statistical variance of mean invasiveness across groups
- Lower = more similar treatment across demographics

### 4. Mean Invasiveness by Race
- Average invasiveness score (0-10) for each racial group
- Based on: medication type, work status, PT, mental health referral, surgery

## Sample Output

```
BASELINE EVALUATION
================================================================================
Baseline Racial Disparity Metrics:
  Disparity Ratio: 1.347
  Gini Coefficient: 0.124
  Variance: 0.456

  Mean Invasiveness by Race:
    White: 4.23
    Black: 5.12  ⚠️ 21% higher
    Hispanic or Latino: 4.89
    Asian: 3.98

GRPO ITERATION 1/10
================================================================================
Iteration 1 Racial Disparity:
  Disparity Ratio: 1.289 (Δ -0.058)
  Gini Coefficient: 0.109 (Δ -0.015)

FINAL EVALUATION
================================================================================
Final Racial Disparity Metrics:
  Disparity Ratio: 1.087  (↓ 19.3%)
  Gini Coefficient: 0.042  (↓ 66.1%)
  Variance: 0.134  (↓ 70.6%)

✅ Training complete!
```

## Output Files

After training:
```
grpo_checkpoints/
├── final_checkpoint.pt         # Final LoRA weights
├── training_history.csv        # Metrics per iteration
└── checkpoint_iter_*.pt        # Intermediate checkpoints
```

## Using Trained Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (use same model you trained with)
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model, 
    "grpo_checkpoints/final_checkpoint.pt"
)

# Generate with reduced bias
prompt = "Clinical vignette here..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Full Workflow Integration

### Step 1: Baseline Experiment
```bash
cd llm_execution/
python run_experiment.py --models llama-3.3-70b --max-vignettes 2304
```

### Step 2: GRPO Training
```bash
cd rl/
./run_grpo_training.sh large
```

### Step 3: Analysis
```bash
cd analysis/
python enhanced_analysis.py
```

Review `grpo_checkpoints/training_history.csv` for:
- Disparity ratio reduction
- Gini coefficient improvement
- Variance decrease

## Technical Details

### GRPO Algorithm

1. **Generate Outputs**: Run model on all vignettes
2. **Calculate Rewards**: Compute racial disparity metrics
3. **Group Adjustment**: Adjust rewards based on group-relative performance
   - Penalize groups with higher-than-average invasiveness
   - Reward groups with lower-than-average invasiveness
4. **Policy Update**: Update model parameters using adjusted rewards
5. **Repeat**: Iterate until disparity is minimized

### LoRA (Low-Rank Adaptation)

- Only trains ~0.1-1% of parameters
- Much faster than full fine-tuning
- Reduces memory requirements
- Can be merged with base model or kept separate

**LoRA Rank Trade-offs:**
- Rank 4-8: Fast, low memory, good for CPU testing
- Rank 16-32: Better quality, requires more memory
- Rank 64+: Best quality, GPU required

### CPU vs GPU

**CPU Mode:**
- ✅ Works for testing and small models
- ✅ No special hardware required
- ❌ Very slow (10-100x slower than GPU)
- ❌ Limited to small models (TinyLlama 1.1B, Phi-2)

**GPU Mode:**
- ✅ Fast training (minutes instead of hours)
- ✅ Can handle large models (7B-70B parameters)
- ✅ Higher LoRA ranks possible
- ❌ Requires NVIDIA GPU with CUDA

**Recommendations:**
- CPU: `--model-size small --num-samples 20 --iterations 2` (TinyLlama 1.1B)
- GPU: Full 2304 samples with 10+ iterations (Llama-2 7B or larger)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size and LoRA rank
./run_grpo_training.sh custom --batch-size 2 --lora-rank 4

# Use smaller model
python train_grpo_nemo.py --model-size small --num-samples 50
```

### Slow Training
```bash
# Use GPU if available
./run_grpo_training.sh medium --device cuda

# Or reduce samples for testing
./run_grpo_training.sh custom --num-samples 50
```

### Setup Issues

**If you see torchvision/torch compatibility errors or "Could not import module 'PreTrainedModel'":**

The issue is that PyTorch 2.9.1 and Transformers 4.57.1 are bleeding-edge versions with compatibility issues.

**COMPLETE FIX (Recommended):**
```bash
# Step 1: Remove problematic versions
pip uninstall torch torchvision torchaudio transformers peft accelerate -y

# Step 2: Install stable PyTorch (CPU)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install compatible transformers/PEFT
pip install transformers==4.44.0 peft==0.12.0 accelerate==0.33.0

# Step 4: Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from peft import LoraConfig, get_peft_model, TaskType; print('✅ PEFT OK')"

# Step 5: Test setup
python test_grpo_setup.py
```

These versions are stable and compatible. They work on CPU machines.

**Other diagnostics:**
```bash
# Run full diagnostic
python test_grpo_setup.py

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check PEFT installation
python -c "from peft import LoraConfig, get_peft_model, TaskType; print('PEFT OK')"
```

### NeMo Import Errors
The script works without NeMo (uses fallback GRPO implementation).

If you see NeMo/torchvision circular import errors:
```bash
# This is OK! The script will use the fallback GRPO implementation
# You can still run training without NeMo:
python train_grpo_nemo.py --model-size small --num-samples 50 --iterations 3
```

To fix NeMo (optional):
```bash
# Reinstall with compatible versions
pip install torchvision==0.20.0 torch==2.5.0
pip install nemo-toolkit[nlp] nemo-aligner

# Use NeMo
python train_grpo_nemo.py --use-nemo
```

### Wandb Authentication Issues

**Error: "permission denied" or "Invalid project name"**

1. **Check your wandb setup:**
```bash
# Login to wandb
wandb login
# Paste your WANDB_API_KEY from .env

# Or set it directly
export WANDB_API_KEY=your_key_here
```

2. **Create the project first:**
   - Go to https://wandb.ai
   - Click "Create Project"
   - Name it "RL" (or your custom name)
   - Set it to "Private" or "Public"

3. **Use correct entity/project:**
```bash
# Check your wandb username at https://wandb.ai/settings
python train_grpo_nemo.py \
  --wandb-entity your_username \
  --wandb-project RL \
  --model-name meta/llama-3.3-70b-instruct
```

4. **Disable wandb (for testing):**
```bash
python train_grpo_nemo.py --no-wandb
```

### HuggingFace Authentication Issues

**Error: "Cannot access gated repo" or "401 Unauthorized"**

1. **Accept model license:**
   - Llama models: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
   - Click "Agree and access repository"

2. **Login with your token:**
```bash
huggingface-cli login
# Paste your HF_TOKEN from .env
```

3. **Verify authentication:**
```bash
huggingface-cli whoami
# Should show your username
```

4. **If still failing, try logout and login again:**
```bash
huggingface-cli logout
huggingface-cli login
```

## Files in This Directory

- **`train_grpo_nemo.py`** - Main GRPO training script (709 lines)
- **`run_grpo_training.sh`** - Convenient wrapper with presets
- **`test_grpo_setup.py`** - Setup verification and quick test
- **`README.md`** - This file

Dependencies are managed in the root `requirements.txt`.

## Key Features

| Feature | Description |
|---------|-------------|
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Framework | NeMo RL compatible (with fallback) |
| Fairness Focus | Racial disparity reduction |
| Metrics | Gini coefficient, disparity ratio, variance |
| LoRA | ✅ Efficient (~1% trainable parameters) |
| CPU Support | ✅ Yes (with performance warnings) |
| GPU Support | ✅ Optimized for CUDA |
| NeMo Integration | ✅ Optional (fallback if unavailable) |

## FAQ

**Q: Do I need a GPU?**
- No, GRPO works on CPU (but 10-100x slower)
- GPU recommended for production

**Q: How long does training take?**
- CPU (small): 10-20 minutes
- GPU (medium): 30-60 minutes
- GPU (large, full factorial): 2-4 hours

**Q: Do I need NeMo installed?**
- No, script uses fallback GRPO if NeMo unavailable
- NeMo optional but recommended for production

**Q: Can I use my own models?**
- Yes! Use `--model-name` with any HuggingFace model
- Example: `--model-name meta/llama-3.3-70b-instruct`

**Q: What's different from standard PPO?**
- GRPO specifically designed for group fairness
- Uses group-relative rewards vs absolute rewards
- Provides interpretable fairness metrics

## Resources

- **NeMo Toolkit**: https://github.com/NVIDIA/NeMo
- **NeMo Aligner**: https://github.com/NVIDIA/NeMo-Aligner
- **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **PEFT Library**: https://github.com/huggingface/peft

## Next Steps

1. ✅ Run `test_grpo_setup.py` to verify installation
2. 📊 Try `./run_grpo_training.sh test` for quick validation
3. 📈 Scale up with `./run_grpo_training.sh medium` or `large`
4. 🔄 Integrate trained checkpoint into your pipeline
5. 📝 Analyze improvements in fairness metrics

---

**Ready to start?** → Run `./run_grpo_training.sh test`
