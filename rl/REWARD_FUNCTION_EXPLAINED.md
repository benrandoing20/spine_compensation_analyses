# REWARD FUNCTION EXPLAINED

## 📊 Overview

The reward function is the **CORE of GRPO training** - it tells the model what behavior to optimize for. In your case: **reducing racial disparities in treatment recommendations**.

---

## 🎯 TWO MODES

### MODE 1: Simple (Equal Weights)
All treatment dimensions weighted equally - good for baseline.

### MODE 2: Weighted (Statistical) ⭐ **RECOMMENDED**
Uses YOUR chi-squared, SHAP, and logistic regression analysis to focus on dimensions that ACTUALLY drive disparity.

---

## 🔍 MODE 1: Simple Reward (Current Default)

```
Reward = -disparity_ratio - (10 × gini) - (0.5 × variance) + (2 × parse_rate)
```

### Components:

1. **Disparity Ratio Penalty** = `-disparity_ratio`
   - Measures max/min invasiveness across races
   - Example: If Black patients get mean invasiveness of 6.0 and White patients get 3.0
     → ratio = 6.0/3.0 = 2.0 → penalty = -2.0
   - Goal: Make ratio → 1.0 (perfect equality)

2. **Gini Penalty** = `-10 × gini_coefficient`
   - Gini = 0 means perfect equality
   - Gini = 1 means perfect inequality
   - Scaled by 10 to make it significant
   - Example: Gini of 0.3 → penalty = -3.0

3. **Variance Penalty** = `-0.5 × variance`
   - Measures spread of mean invasiveness across racial groups
   - Lower variance = more consistent treatment
   - Example: variance of 2.0 → penalty = -1.0

4. **Parse Bonus** = `2 × parse_rate`
   - Rewards valid JSON outputs
   - Example: 90% valid → bonus = +1.8

**Total Reward Example:**
```
disparity_ratio = 2.0  → penalty = -2.0
gini = 0.3             → penalty = -3.0
variance = 2.0         → penalty = -1.0
parse_rate = 0.9       → bonus = +1.8
─────────────────────────────────────
TOTAL REWARD = -4.2
```

### ❌ Problem with Simple Approach:
- **Treats all treatment dimensions equally**
- But YOUR analysis showed some dimensions have MUCH bigger impact on disparity!
- Example: Maybe medication choice drives 40% of disparity, but PT only 5%
- Simple approach can't focus on what matters most

---

## 🎓 MODE 2: Weighted Reward (Using Your Statistical Analysis)

### Your Statistical Analysis (What You've Done):

1. **Chi-Squared Test**
   - Tests if treatment choice is independent of race
   - Shows which dimensions have significant racial disparities
   - Example results:
     ```
     Medication:      χ² = 45.2, p < 0.001  (HIGHLY significant)
     Surgical:        χ² = 32.1, p < 0.001  (HIGHLY significant)
     Work status:     χ² = 18.4, p < 0.01   (Significant)
     Mental health:   χ² = 8.2,  p < 0.05   (Marginally significant)
     Physical therapy: χ² = 2.1,  p = 0.15   (NOT significant)
     ```

2. **SHAP Values (Feature Importance)**
   - Shows which features contribute MOST to predicting disparate outcomes
   - Example SHAP importance:
     ```
     Medication:       SHAP = 0.35  (35% of total importance)
     Surgical:         SHAP = 0.25  (25%)
     Work status:      SHAP = 0.20  (20%)
     Mental health:    SHAP = 0.15  (15%)
     Physical therapy: SHAP = 0.05  (5%)
     ```

3. **Logistic Regression (Controlled Effects)**
   - Controls for all other factors to isolate each dimension's independent effect
   - Shows the CAUSAL impact on disparity
   - Example coefficients (predicting high invasiveness):
     ```
     Medication:       β = 0.82, OR = 2.27, p < 0.001
     Surgical:         β = 0.64, OR = 1.90, p < 0.001
     Work status:      β = 0.45, OR = 1.57, p < 0.01
     Mental health:    β = 0.28, OR = 1.32, p < 0.05
     Physical therapy: β = 0.12, OR = 1.13, p = 0.12 (not significant)
     ```

### How to Use This in the Reward Function:

**STEP 1: Convert your statistical findings to weights**

Based on combined evidence from chi-squared, SHAP, and logistic regression:

```python
dimension_weights = {
    'medication': 0.40,      # Highest chi², SHAP, and β
    'work_status': 0.25,     # Medium across all tests
    'surgical': 0.20,        # High impact but less frequent
    'mental_health': 0.10,   # Lower but still significant
    'physical_therapy': 0.05 # Minimal disparity observed
}
# Total = 1.0
```

**STEP 2: Compute disparity PER DIMENSION**

Instead of one overall disparity score, compute separately:

```python
medication_disparity = max_race_med / min_race_med
surgical_disparity = max_race_surg / min_race_surg
... etc
```

**STEP 3: Weight by importance**

```python
weighted_disparity = (
    0.40 × medication_disparity +
    0.25 × work_status_disparity +
    0.20 × surgical_disparity +
    0.10 × mental_health_disparity +
    0.05 × pt_disparity
)
```

**STEP 4: Total weighted reward**

```python
Reward = -weighted_disparity + gini_penalty + parse_bonus + quality_penalty
```

---

## 📈 Comparison Example

**Scenario**: Model generates recommendations for 2,304 vignettes

### Outcome A (Simple reward):
```
Overall disparity ratio: 2.0
Medication disparity: 3.5  (very high!)
PT disparity: 1.1  (low)
Work status disparity: 1.8

Simple reward: -2.0 - (10 × 0.3) - (0.5 × 2.0) + 1.8 = -4.2
```

### Outcome B (Weighted reward):
```
Medication disparity: 3.5
Work status disparity: 1.8
Surgical disparity: 2.2
Mental health disparity: 1.5
PT disparity: 1.1

Weighted disparity = 0.40×3.5 + 0.25×1.8 + 0.20×2.2 + 0.10×1.5 + 0.05×1.1
                   = 1.40 + 0.45 + 0.44 + 0.15 + 0.055
                   = 2.495

Weighted reward: -2.495 - (5 × 0.3) + 1.8 + 0 = -2.195
```

**Why weighted is better:**
- Focuses optimization on **medication** (40% weight, 3.5 disparity)
- Less concerned about **PT** (5% weight, 1.1 disparity)
- Aligns with YOUR statistical findings about what drives disparity

---

## 🛡️ Clinical Quality Constraints

Problem: Model might "game" the reward by recommending nothing to everyone → zero disparity but terrible care!

Solution: Add clinical quality penalties:

```python
- Too many opioids (>30%)        → penalty
- Too many surgeries (>20%)      → penalty  
- Too few PT orders (<50%)       → penalty
- Extreme work restrictions (>40% off work) → penalty
```

This ensures model learns **equitable AND appropriate** care.

---

## 🔧 How to Update the Code

**Option 1: Update weights directly in code** (Lines 297-303)

```python
dimension_weights = {
    'medication': 0.40,      # ← UPDATE with your SHAP/logistic β
    'work_status': 0.25,     # ← UPDATE
    'surgical': 0.20,        # ← UPDATE
    'mental_health': 0.10,   # ← UPDATE
    'physical_therapy': 0.05 # ← UPDATE
}
```

**Option 2: Pass weights as config**

Create a JSON file with your analysis results:

```json
{
  "dimension_weights": {
    "medication": 0.42,
    "work_status": 0.23,
    "surgical": 0.21,
    "mental_health": 0.09,
    "physical_therapy": 0.05
  },
  "logistic_coefficients": {
    "medication": {"beta": 0.82, "p_value": 0.001},
    "surgical": {"beta": 0.64, "p_value": 0.001},
    ...
  }
}
```

---

## 📊 Wandb Tracking

Both reward modes log to Wandb:

**Simple mode logs:**
- `reward/total`
- `disparity/ratio`
- `disparity/gini`
- `disparity/variance`

**Weighted mode ALSO logs:**
- `disparity/medication`
- `disparity/work_status`
- `disparity/surgical`
- `disparity/mental_health`
- `disparity/physical_therapy`
- `clinical_quality/opioid_rate`
- `clinical_quality/surgery_rate`
- `clinical_quality/pt_rate`
- `clinical_quality/off_work_rate`

This lets you see which dimensions improve during training!

---

## 🚀 Recommendation

**START with simple mode** (test run):
```bash
python train_grpo_nemo.py \
  --model-name qwen/qwen3-next-80b-a3b-instruct \
  --num-samples 100 \
  --iterations 2
```

**THEN switch to weighted mode** with your actual analysis results:

1. Extract weights from your chi-squared, SHAP, logistic regression
2. Update lines 297-303 in train_grpo_nemo.py
3. Run full training:

```bash
python train_grpo_nemo.py \
  --model-name qwen/qwen3-next-80b-a3b-instruct \
  --num-samples 2304 \
  --iterations 10
```

---

## 📚 References

- **GRPO Paper**: Uses group-based rewards to optimize for fairness
- **Your Analysis**: Chi-squared + SHAP + Logistic regression identified key drivers
- **Clinical Guidelines**: Inform quality constraints

The weighted approach is MORE PRINCIPLED because it's based on YOUR EMPIRICAL FINDINGS about what causes disparity!
