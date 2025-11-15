#!/usr/bin/env python3
"""
Enhanced analysis with odds ratios, counts/percentages, and average marginal effects.

Usage:
    python enhanced_analysis.py results/merged_results.csv
    python enhanced_analysis.py results/merged_results.csv --output enhanced_output/
"""

import argparse
import sys
from pathlib import Path
import warnings
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# Setup
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def setup_logging():
    """Setup logging to save to logs/enhanced_analysis/ directory"""
    # Create logs directory structure
    log_dir = Path('logs/enhanced_analysis')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'enhanced_analysis_{timestamp}.log'
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def load_and_separate(csv_path):
    """Load merged results and separate baseline from full experiment."""
    df = pd.read_csv(csv_path)
    
    # Separate baseline and full results
    baseline = df[df['model'].str.startswith('baseline-')].copy()
    full = df[~df['model'].str.startswith('baseline-')].copy()
    
    # Add clean model name to baseline (without 'baseline-' prefix)
    baseline['base_model'] = baseline['model'].str.replace('baseline-', '')
    
    logging.info(f"Loaded {len(df)} total results:")
    logging.info(f"  - Baseline: {len(baseline)} ({len(baseline['model'].unique())} models)")
    logging.info(f"  - Full experiment: {len(full)} ({len(full['model'].unique())} models)")
    
    return baseline, full


def calculate_odds_ratio_table(full, baseline, outcome_col, demographic_col, model_name):
    """
    Calculate odds ratios comparing ALL pairs of demographic groups (full factorial).
    For example: Black vs White, Black vs Asian, White vs Asian, etc.
    Returns a table with odds ratios, confidence intervals, and p-values.
    """
    # Filter to specific model
    full_model = full[full['model'] == model_name]
    
    # Get unique outcome values and demographic values
    outcome_values = sorted(full_model[outcome_col].dropna().unique())
    demographic_values = sorted(full_model[demographic_col].dropna().unique())
    
    if len(demographic_values) < 2:
        return None
    
    results = []
    
    # Compare all pairs of demographic values (full factorial)
    for outcome_value in outcome_values:
        for i, reference_value in enumerate(demographic_values):
            # Reference group counts
            ref_data = full_model[full_model[demographic_col] == reference_value]
            ref_outcome_count = (ref_data[outcome_col] == outcome_value).sum()
            ref_total = len(ref_data)
            ref_pct = ref_outcome_count / ref_total if ref_total > 0 else 0
            
            # Compare to all other demographics (only those after this one to avoid duplicates)
            for demo_value in demographic_values[i+1:]:
                demo_data = full_model[full_model[demographic_col] == demo_value]
                
                # Demographic counts
                demo_outcome_count = (demo_data[outcome_col] == outcome_value).sum()
                demo_total = len(demo_data)
                demo_pct = demo_outcome_count / demo_total if demo_total > 0 else 0
                
                # Calculate odds ratio
                # Odds of outcome in demographic group
                demo_odds = demo_outcome_count / (demo_total - demo_outcome_count) if (demo_total - demo_outcome_count) > 0 else np.inf
                
                # Odds of outcome in reference group
                ref_odds = ref_outcome_count / (ref_total - ref_outcome_count) if (ref_total - ref_outcome_count) > 0 else np.inf
                
                # Odds ratio
                odds_ratio = demo_odds / ref_odds if ref_odds > 0 and ref_odds != np.inf and demo_odds != np.inf else np.nan
                
                # 95% Confidence interval (using log transformation)
                # Standard error of log(OR)
                if (demo_outcome_count > 0 and (demo_total - demo_outcome_count) > 0 and 
                    ref_outcome_count > 0 and (ref_total - ref_outcome_count) > 0):
                    
                    se_log_or = np.sqrt(
                        1/demo_outcome_count + 
                        1/(demo_total - demo_outcome_count) + 
                        1/ref_outcome_count + 
                        1/(ref_total - ref_outcome_count)
                    )
                    
                    log_or = np.log(odds_ratio)
                    ci_lower = np.exp(log_or - 1.96 * se_log_or)
                    ci_upper = np.exp(log_or + 1.96 * se_log_or)
                else:
                    ci_lower = np.nan
                    ci_upper = np.nan
                
                # Chi-square test
                contingency = np.array([
                    [demo_outcome_count, demo_total - demo_outcome_count],
                    [ref_outcome_count, ref_total - ref_outcome_count]
                ])
                
                try:
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                except:
                    chi2, p_value = np.nan, np.nan
                
                # Interpretation
                if not np.isnan(odds_ratio):
                    if odds_ratio > 1:
                        pct_change = (odds_ratio - 1) * 100
                        interpretation = f"{pct_change:.1f}% more likely than {reference_value}"
                    elif odds_ratio < 1:
                        pct_change = (1 - odds_ratio) * 100
                        interpretation = f"{pct_change:.1f}% less likely than {reference_value}"
                    else:
                        interpretation = f"Same as {reference_value}"
                else:
                    interpretation = "Cannot calculate"
                
                results.append({
                    'model': model_name,
                    'outcome': outcome_col,
                    'outcome_value': outcome_value,
                    'demographic': demographic_col,
                    'reference_group': reference_value,
                    'comparison_group': demo_value,
                    'reference_count': ref_outcome_count,
                    'reference_total': ref_total,
                    'reference_pct': ref_pct * 100,
                    'comparison_count': demo_outcome_count,
                    'comparison_total': demo_total,
                    'comparison_pct': demo_pct * 100,
                    'odds_ratio': odds_ratio,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                    'interpretation': interpretation
                })
    
    return pd.DataFrame(results)


def medication_prescription_counts_table(full, baseline, demographic_col, model_name):
    """
    DEPRECATED: Compares medication patterns to baseline (too small sample).
    Use medication_prescription_group_comparisons() instead.
    
    This function is kept for backward compatibility but produces uninformative results
    because the baseline sample is too small (typically n=3).
    """
    outcome_col = 'Medication prescription'
    
    # Filter to specific model
    full_model = full[full['model'] == model_name]
    baseline_model = baseline[baseline['base_model'] == model_name]
    
    if len(baseline_model) == 0:
        return None
    
    # Get all medication types
    medication_types = sorted(full_model[outcome_col].dropna().unique())
    demographic_values = sorted(full_model[demographic_col].dropna().unique())
    
    results = []
    
    # Baseline counts
    baseline_total = len(baseline_model)
    baseline_counts = baseline_model[outcome_col].value_counts()
    
    for demo_value in demographic_values:
        demo_data = full_model[full_model[demographic_col] == demo_value]
        demo_total = len(demo_data)
        demo_counts = demo_data[outcome_col].value_counts()
        
        # Chi-square test between demographic group and baseline
        all_meds = sorted(set(baseline_counts.index) | set(demo_counts.index))
        contingency = np.array([
            [demo_counts.get(med, 0) for med in all_meds],
            [baseline_counts.get(med, 0) for med in all_meds]
        ])
        
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        except:
            chi2, p_value = np.nan, np.nan
        
        # Build count string for each medication
        med_strings = []
        for med in medication_types:
            demo_count = demo_counts.get(med, 0)
            demo_pct = (demo_count / demo_total * 100) if demo_total > 0 else 0
            baseline_count = baseline_counts.get(med, 0)
            baseline_pct = (baseline_count / baseline_total * 100) if baseline_total > 0 else 0
            
            med_strings.append(f"{med}: {demo_count} ({demo_pct:.1f}%) vs baseline {baseline_count} ({baseline_pct:.1f}%)")
        
        results.append({
            'model': model_name,
            'demographic': demographic_col,
            'demographic_value': demo_value,
            'total_n': demo_total,
            'medication_counts': '; '.join(med_strings),
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    return pd.DataFrame(results)


def medication_prescription_group_comparisons(full, demographic_col, model_name):
    """
    Compare medication prescription patterns between demographic groups (group vs group).
    
    This is the PREFERRED method as it compares meaningful groups to each other
    (e.g., old vs young, Black vs White, etc.) rather than to a tiny baseline sample.
    
    Performs pairwise chi-square tests for all combinations of demographic groups.
    """
    outcome_col = 'Medication prescription'
    
    # Filter to specific model
    full_model = full[full['model'] == model_name]
    
    if outcome_col not in full_model.columns or demographic_col not in full_model.columns:
        return None
    
    # Get all demographic values
    demographic_values = sorted(full_model[demographic_col].dropna().unique())
    
    if len(demographic_values) < 2:
        return None
    
    results = []
    
    # Compare each pair of demographic groups
    for i, group1 in enumerate(demographic_values):
        for group2 in demographic_values[i+1:]:
            # Get data for each group
            group1_data = full_model[full_model[demographic_col] == group1]
            group2_data = full_model[full_model[demographic_col] == group2]
            
            group1_total = len(group1_data)
            group2_total = len(group2_data)
            
            # Get medication counts
            group1_counts = group1_data[outcome_col].value_counts()
            group2_counts = group2_data[outcome_col].value_counts()
            
            # Get all medication types present in either group
            all_meds = sorted(set(group1_counts.index) | set(group2_counts.index))
            
            # Build contingency table
            contingency = np.array([
                [group1_counts.get(med, 0) for med in all_meds],
                [group2_counts.get(med, 0) for med in all_meds]
            ])
            
            # Chi-square test
            try:
                chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            except:
                chi2, p_value, dof = np.nan, np.nan, np.nan
                
            # Build detailed medication string
            med_details = []
            for med in all_meds:
                g1_count = group1_counts.get(med, 0)
                g1_pct = (g1_count / group1_total * 100) if group1_total > 0 else 0
                g2_count = group2_counts.get(med, 0)
                g2_pct = (g2_count / group2_total * 100) if group2_total > 0 else 0
                med_details.append(f"{med}: {g1_count}({g1_pct:.1f}%) vs {g2_count}({g2_pct:.1f}%)")
            
            results.append({
                'model': model_name,
                'demographic': demographic_col,
                'group1': group1,
                'group2': group2,
                'group1_n': group1_total,
                'group2_n': group2_total,
                'chi2': chi2,
                'dof': dof,
                'p_value': p_value,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                'medication_details': ' | '.join(med_details)
            })
    
    return pd.DataFrame(results)


def calculate_average_marginal_effects(full, outcome_col, demographic_col, model_name, outcome_value):
    """
    Calculate Average Marginal Effects (AME) for ALL pairs of demographic categories (full factorial).
    
    AME represents the average change in probability of the outcome across the sample
    when changing from one category to another.
    """
    # Filter to specific model
    full_model = full[full['model'] == model_name].copy()
    
    # Remove any rows with missing values
    full_model = full_model[[outcome_col, demographic_col]].dropna()
    
    # Create binary outcome
    full_model['outcome_binary'] = (full_model[outcome_col] == outcome_value).astype(int)
    
    # Get all demographic categories
    demographic_values = sorted(full_model[demographic_col].unique())
    
    if len(demographic_values) < 2:
        return None
    
    # Create dummy variables for demographic
    demo_dummies = pd.get_dummies(full_model[demographic_col], prefix=demographic_col, drop_first=False)
    
    # Use first category as reference (drop it) for the model
    reference_category = demo_dummies.columns[0]
    X = demo_dummies.drop(columns=[reference_category]).astype(float)
    
    # Add constant
    X = add_constant(X, has_constant='add')
    
    y = full_model['outcome_binary'].astype(float)
    
    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Drop any remaining NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(y) < 10:
        return None
    
    # Fit logistic regression
    try:
        model = Logit(y, X)
        result = model.fit(disp=0, maxiter=100)
        
        # Calculate AME for all pairwise comparisons
        ame_results = []
        
        # For each pair of demographic categories
        for i, ref_cat in enumerate(demographic_values):
            for comp_cat in demographic_values[i+1:]:
                # Create datasets representing each category
                X_ref = X.copy()
                X_comp = X.copy()
                
                # Set all demographic dummies to 0
                for c in X.columns:
                    if c.startswith(demographic_col):
                        X_ref[c] = 0.0
                        X_comp[c] = 0.0
                
                # Set the appropriate dummy to 1 for each
                ref_col = f"{demographic_col}_{ref_cat}"
                comp_col = f"{demographic_col}_{comp_cat}"
                
                # If ref_cat is the reference (not in X), keep it at 0
                if ref_col in X.columns:
                    X_ref[ref_col] = 1.0
                    
                # If comp_cat is the reference (not in X), keep it at 0
                if comp_col in X.columns:
                    X_comp[comp_col] = 1.0
                
                # Predict probabilities
                prob_ref = result.predict(X_ref)
                prob_comp = result.predict(X_comp)
                
                # AME is the average difference
                ame = (prob_comp - prob_ref).mean()
                
                # For p-value: if comparing to reference category, use direct p-value
                # Otherwise, need to test difference (use approximation)
                if ref_cat == reference_category.replace(f"{demographic_col}_", "") and comp_col in X.columns:
                    p_value = result.pvalues[comp_col]
                    ame_se = result.bse[comp_col] * prob_comp.mean() * (1 - prob_comp.mean())
                elif comp_cat == reference_category.replace(f"{demographic_col}_", "") and ref_col in X.columns:
                    p_value = result.pvalues[ref_col]
                    ame_se = result.bse[ref_col] * prob_ref.mean() * (1 - prob_ref.mean())
                else:
                    # Both are non-reference categories - approximate with Wald test
                    # This is a simplified approximation
                    if ref_col in X.columns and comp_col in X.columns:
                        se_ref = result.bse[ref_col]
                        se_comp = result.bse[comp_col]
                        # Approximate SE of difference
                        ame_se = np.sqrt(se_ref**2 + se_comp**2) * prob_comp.mean() * (1 - prob_comp.mean())
                        # Approximate z-test
                        z_stat = ame / (ame_se + 1e-10)
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    else:
                        ame_se = np.nan
                        p_value = np.nan
                
                # Confidence interval
                ame_ci_lower = ame - 1.96 * ame_se if not np.isnan(ame_se) else np.nan
                ame_ci_upper = ame + 1.96 * ame_se if not np.isnan(ame_se) else np.nan
                
                ame_results.append({
                    'model': model_name,
                    'outcome': outcome_col,
                    'outcome_value': outcome_value,
                    'demographic': demographic_col,
                    'reference_category': ref_cat,
                    'demographic_value': comp_cat,
                    'AME': ame,
                    'AME_se': ame_se,
                    'AME_ci_lower': ame_ci_lower,
                    'AME_ci_upper': ame_ci_upper,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                    'interpretation': f"{ame * 100:+.1f} percentage point change vs reference"
                })
        
        return pd.DataFrame(ame_results)
        
    except Exception as e:
        logging.warning(f"  ⚠️  Could not calculate AME for {outcome_col} × {demographic_col}: {e}")
        return None


def train_logistic_regression_models(full, model_name):
    """
    Train logistic regression models for each outcome using all demographics as features.
    Uses hyperparameter search to find the best model, then returns coefficient tables.
    
    Baseline categories (dropped to avoid multicollinearity):
    - white (race_ethnicity)
    - cis man (gender_identity)
    - hetero (sexual_orientation)
    - upperclass (socioeconomic_status)
    - whitecollar (occupation_type)
    - english proficient (language_proficiency)
    - urban (geography)
    - young (age_band)
    """
    # Define baseline categories to drop (must match exact values in data)
    baseline_categories = {
        'race_ethnicity': 'White',
        'gender_identity': 'cisgender man',
        'sexual_orientation': 'heterosexual',
        'socioeconomic_status': 'upper class',
        'occupation_type': 'white collar',
        'language_proficiency': 'English proficient',
        'geography': 'urban',
        'age_band': 'young'
    }
    
    # Filter to specific model
    full_model = full[full['model'] == model_name].copy()
    
    # Define all outcomes to model
    outcomes = [
        'Medication prescription',
        'work_status',
        'mental_health_referral',
        'physical_therapy',
        'surgical_referral'
    ]
    
    # Check for TTD duration column (it may have different name)
    ttd_col_options = [
        'ttd_duration_weeks',
        'If Off work/Temporary Total Disability, duration in weeks',
        'TTD_duration_weeks'
    ]
    for ttd_col in ttd_col_options:
        if ttd_col in full_model.columns:
            outcomes.append(ttd_col)
            break
    
    demographics = ['age_band', 'race_ethnicity', 'gender_identity', 
                   'sexual_orientation', 'socioeconomic_status', 'occupation_type',
                   'language_proficiency', 'geography']
    
    results = {}
    
    for outcome in outcomes:
        if outcome not in full_model.columns:
            logging.warning(f"  ⚠️  Outcome '{outcome}' not found in data, skipping...")
            continue
        
        logging.info(f"\n  Training logistic regression for: {outcome}")
        
        # Prepare data: demographics + outcome
        data = full_model[demographics + [outcome]].copy()
        
        # Drop rows with any missing values
        data = data.dropna()
        
        if len(data) < 50:
            logging.warning(f"    ⚠️  Insufficient data ({len(data)} rows), skipping...")
            continue
        
        # For TTD duration, convert to binary (0 weeks vs >0 weeks)
        if 'duration' in outcome.lower() or 'ttd' in outcome.lower():
            # Convert to numeric if it's not already
            data[outcome] = pd.to_numeric(data[outcome], errors='coerce')
            data = data.dropna(subset=[outcome])
            data[outcome + '_binary'] = (data[outcome] > 0).astype(int)
            y = data[outcome + '_binary']
        else:
            y = data[outcome]
        
        # Create dummy variables for all demographics
        X = pd.DataFrame()
        
        for demo in demographics:
            # Get dummies
            dummies = pd.get_dummies(data[demo], prefix=demo, drop_first=False)
            
            # Drop the baseline category if it exists
            baseline_col = f"{demo}_{baseline_categories[demo]}"
            if baseline_col in dummies.columns:
                dummies = dummies.drop(columns=[baseline_col])
            
            X = pd.concat([X, dummies], axis=1)
        
        # Align X and y indices
        X = X.loc[y.index]
        
        # For multi-class outcomes, convert to binary by selecting ONE target class
        # This ensures we get ONE row per model per outcome
        unique_classes = y.unique()
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            logging.warning(f"    ⚠️  Only one class present, skipping...")
            continue
        
        if n_classes > 2:
            # Multi-class: pick the most clinically relevant or common positive class
            # For medication: prefer more aggressive treatment
            # For other outcomes: prefer the positive/intervention class
            target_class_priority = {
                'Medication prescription': ['Opioid', 'Steroid injection', 'Oral Steroid', 'Prescription non-opioid', 'OTC only'],
                'mental_health_referral': ['Formal psych/mental health evaluation', 'Optional counseling', 'No referral'],
                'work_status': ['Off work/Temporary Total Disability', 'Modified duty', 'Full duty']
            }
            
            # Get priority list for this outcome, or just use alphabetical
            priority = target_class_priority.get(outcome, sorted(unique_classes, reverse=True))
            
            # Find first class in priority list that exists in data
            target_class = None
            for candidate in priority:
                if candidate in unique_classes:
                    target_class = candidate
                    break
            
            if target_class is None:
                target_class = unique_classes[0]
            
            # Convert to binary: target_class vs all others
            y = (y == target_class).astype(int)
            logging.info(f"    Multi-class outcome detected. Modeling: '{target_class}' vs others")
        
        # Encode target (now guaranteed to be binary)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        n_classes = len(np.unique(y_encoded))
        
        if n_classes < 2:
            logging.warning(f"    ⚠️  Only one class present after conversion, skipping...")
            continue
        
        # Hyperparameter search (binary classification only now)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [1000]
        }
        
        logging.info(f"    Running hyperparameter search (binary classification)...")
        
        # Grid search with cross-validation
        try:
            lr = LogisticRegression(random_state=42)
            grid_search = GridSearchCV(
                lr, 
                param_grid, 
                cv=min(5, len(y) // 20),  # Adaptive CV folds
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y_encoded)
            
            # Best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logging.info(f"    ✓ Best params: {best_params}")
            logging.info(f"    ✓ Best CV accuracy: {best_score:.4f}")
            
            # Extract coefficients (binary classification)
            coefs = pd.DataFrame({
                'feature': X.columns,
                'coefficient': best_model.coef_[0],
                'odds_ratio': np.exp(best_model.coef_[0])
            })
            coefs['intercept'] = best_model.intercept_[0]
            
            # Add metadata
            coefs['model'] = model_name
            coefs['outcome'] = outcome
            coefs['n_samples'] = len(y)
            coefs['n_features'] = len(X.columns)
            coefs['cv_accuracy'] = best_score
            coefs['C'] = best_params['C']
            
            # Sort by absolute coefficient value
            coefs['abs_coef'] = coefs['coefficient'].abs()
            coefs = coefs.sort_values('abs_coef', ascending=False)
            coefs = coefs.drop(columns=['abs_coef'])
            
            results[outcome] = coefs
            
            logging.info(f"    ✓ Extracted {len(coefs)} coefficients")
            
        except Exception as e:
            logging.warning(f"    ⚠️  Error training model: {e}")
            continue
    
    return results


def create_formatted_logistic_regression_tables(all_lr_results, output_dir, models_list):
    """
    Create simple pivot tables for logistic regression results.
    ONE table per outcome showing coefficients organized by model (rows) and features (columns).
    All outcomes are binary (multi-class converted to binary in training).
    Also creates a master combined table with all outcomes stacked.
    """
    if not all_lr_results:
        return
    
    # Combine all results
    lr_combined = pd.concat(all_lr_results, ignore_index=True)
    
    # Get unique outcomes
    outcomes = lr_combined['outcome'].unique()
    
    # Storage for combined table
    all_pivots = []
    
    for outcome in outcomes:
        outcome_data = lr_combined[lr_combined['outcome'] == outcome].copy()
        
        logging.info(f"\n{'='*80}")
        logging.info(f"LOGISTIC REGRESSION COEFFICIENTS: {outcome}")
        logging.info(f"{'='*80}")
        logging.info(f"Reference categories (dropped): White, cisgender man, heterosexual, upper class,")
        logging.info(f"                                 white collar, English proficient, urban, young")
        logging.info("")
        
        # Create pivot table: models as rows, features as columns, coefficients as values
        pivot = outcome_data.pivot_table(
            index='model',
            columns='feature',
            values='coefficient',
            aggfunc='first'
        )
        
        # Add outcome column for the combined table
        pivot_with_outcome = pivot.copy()
        pivot_with_outcome.insert(0, 'outcome', outcome)
        all_pivots.append(pivot_with_outcome)
        
        # Save individual outcome table
        safe_outcome_name = outcome.replace(' ', '_').replace('/', '_').replace(',', '').lower()
        output_file = output_dir / f'logistic_regression_coef_{safe_outcome_name}.csv'
        pivot.to_csv(output_file)
        
        logging.info(f"Table saved to: {output_file}")
        logging.info(f"Dimensions: {pivot.shape[0]} models × {pivot.shape[1]} features")
        
        # Show which models were trained
        trained_models = list(pivot.index)
        logging.info(f"Models with data: {', '.join(trained_models)}")
        missing_models = set(models_list) - set(trained_models)
        if missing_models:
            logging.info(f"Models unable to train (no variation in outcome): {', '.join(missing_models)}")
        logging.info("")
    
    # Create master combined table
    if all_pivots:
        logging.info(f"\n{'='*80}")
        logging.info(f"CREATING MASTER COMBINED TABLE")
        logging.info(f"{'='*80}\n")
        
        # Stack all tables
        master_table = pd.concat(all_pivots, axis=0)
        
        # Save master table
        master_output = output_dir / 'logistic_regression_coef_ALL_OUTCOMES.csv'
        master_table.to_csv(master_output)
        
        logging.info(f"✅ Master table saved to: {master_output}")
        logging.info(f"   Dimensions: {master_table.shape[0]} total rows × {master_table.shape[1]} columns")
        logging.info(f"   Contains all {len(outcomes)} outcomes stacked together")
        logging.info("")
        
        # Show breakdown by outcome
        outcome_counts = master_table.groupby('outcome').size()
        logging.info("Breakdown by outcome:")
        for outcome, count in outcome_counts.items():
            logging.info(f"  - {outcome}: {count} models")
        logging.info("")


def main():
    parser = argparse.ArgumentParser(description="Enhanced analysis with odds ratios and AME")
    parser.add_argument('csv_file', help='Path to merged results CSV')
    parser.add_argument('--output', default='analysis/enhanced_output', help='Output directory')
    parser.add_argument('--model', help='Specific model to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info("\n" + "=" * 80)
    logging.info("ENHANCED ANALYSIS: Odds Ratios, Counts, and Average Marginal Effects")
    logging.info("=" * 80 + "\n")
    logging.info(f"Logging to: {log_file}\n")
    
    baseline, full = load_and_separate(args.csv_file)
    
    # Define outcomes and demographics to analyze
    outcomes = ['surgical_referral', 'work_status', 'mental_health_referral', 'physical_therapy']
    
    demographics = ['age_band', 'race_ethnicity', 'gender_identity', 
                   'sexual_orientation', 'socioeconomic_status', 'occupation_type',
                   'language_proficiency', 'geography']
    
    # Get models to analyze
    if args.model:
        models = [args.model]
    else:
        models = full['model'].unique()
    
    logging.info(f"Analyzing {len(models)} models: {list(models)}\n")
    
    # Storage for all results
    all_odds_ratios = []
    all_medication_counts = []  # DEPRECATED: baseline comparison
    all_medication_group_comparisons = []  # NEW: group-to-group comparison
    all_ame = []
    all_logistic_regression = []  # NEW: Logistic regression coefficients
    
    # Run analysis for each model
    for model in models:
        logging.info(f"\n{'='*80}")
        logging.info(f"MODEL: {model}")
        logging.info(f"{'='*80}\n")
        
        # ===== ODDS RATIOS =====
        logging.info("📊 Calculating Odds Ratios...")
        for outcome in outcomes:
            if outcome not in full.columns:
                continue
            
            for demographic in demographics:
                if demographic not in full.columns:
                    continue
                
                or_df = calculate_odds_ratio_table(full, baseline, outcome, demographic, model)
                
                if or_df is not None and len(or_df) > 0:
                    all_odds_ratios.append(or_df)
                    
                    # Print significant findings
                    sig = or_df[or_df['significant']]
                    if len(sig) > 0:
                        logging.info(f"\n  ✓ {outcome} × {demographic} (significant findings):")
                        for _, row in sig.iterrows():
                            logging.info(f"    • {row['comparison_group']} vs {row['reference_group']} → {row['outcome_value']}: "
                                 f"OR={row['odds_ratio']:.2f} (95% CI: {row['ci_lower']:.2f}-{row['ci_upper']:.2f}), "
                                 f"p={row['p_value']:.4f} | {row['interpretation']}")
        
        # ===== MEDICATION PRESCRIPTION GROUP COMPARISONS =====
        logging.info("\n\n💊 Medication Prescription Group Comparisons (Group vs Group)...")
        for demographic in demographics:
            if demographic not in full.columns:
                continue
            
            med_df = medication_prescription_group_comparisons(full, demographic, model)
            
            if med_df is not None and len(med_df) > 0:
                all_medication_group_comparisons.append(med_df)
                
                # Print significant findings
                sig = med_df[med_df['significant']]
                if len(sig) > 0:
                    logging.info(f"\n  ✓ {demographic} (significant differences between groups):")
                    for _, row in sig.iterrows():
                        logging.info(f"    • {row['group1']} vs {row['group2']}: "
                             f"χ²={row['chi2']:.2f}, p={row['p_value']:.4f}")
        
        # ===== AVERAGE MARGINAL EFFECTS =====
        logging.info("\n\n📈 Calculating Average Marginal Effects (AME)...")
        for outcome in outcomes:
            if outcome not in full.columns:
                continue
            
            # Get all unique outcome values (like odds ratios does)
            full_model = full[full['model'] == model]
            outcome_values = sorted(full_model[outcome].dropna().unique())
            
            # Calculate AME for each outcome value
            for outcome_value in outcome_values:
                for demographic in demographics:
                    if demographic not in full.columns:
                        continue
                    
                    ame_df = calculate_average_marginal_effects(full, outcome, demographic, model, outcome_value)
                    
                    if ame_df is not None and len(ame_df) > 0:
                        all_ame.append(ame_df)
                        
                        # Print significant findings
                        sig = ame_df[ame_df['significant']]
                        if len(sig) > 0:
                            logging.info(f"\n  ✓ {outcome} = '{outcome_value}' × {demographic}:")
                            for _, row in sig.iterrows():
                                logging.info(f"    • {row['demographic_value']} (vs {row['reference_category']}): "
                                     f"AME={row['AME']:.4f} ({row['AME_ci_lower']:.4f} to {row['AME_ci_upper']:.4f}), "
                                     f"p={row['p_value']:.4f} | {row['interpretation']}")
        
        # ===== LOGISTIC REGRESSION WITH HYPERPARAMETER TUNING =====
        logging.info("\n\n🤖 Training Logistic Regression Models (with hyperparameter search)...")
        lr_results = train_logistic_regression_models(full, model)
        
        if lr_results:
            for outcome, coef_df in lr_results.items():
                all_logistic_regression.append(coef_df)
                logging.info(f"\n  ✓ {outcome}: Trained successfully")
    
    # ===== SAVE ALL RESULTS =====
    logging.info("\n\n" + "=" * 80)
    logging.info("SAVING RESULTS")
    logging.info("=" * 80)
    
    if all_odds_ratios:
        or_combined = pd.concat(all_odds_ratios, ignore_index=True)
        or_output = output_dir / 'odds_ratios_all.csv'
        or_combined.to_csv(or_output, index=False)
        logging.info(f"\n✅ Odds ratios saved: {or_output}")
        logging.info(f"   Total comparisons: {len(or_combined)}")
        logging.info(f"   Significant: {or_combined['significant'].sum()} ({or_combined['significant'].mean()*100:.1f}%)")
        
        # Save significant only
        or_sig = or_combined[or_combined['significant']]
        or_sig_output = output_dir / 'odds_ratios_significant.csv'
        or_sig.to_csv(or_sig_output, index=False)
        logging.info(f"   Significant only: {or_sig_output}")
    
    if all_medication_group_comparisons:
        med_combined = pd.concat(all_medication_group_comparisons, ignore_index=True)
        med_output = output_dir / 'medication_group_comparisons_all.csv'
        med_combined.to_csv(med_output, index=False)
        logging.info(f"\n✅ Medication group comparisons saved: {med_output}")
        logging.info(f"   Total comparisons: {len(med_combined)}")
        logging.info(f"   Significant: {med_combined['significant'].sum()} ({med_combined['significant'].mean()*100:.1f}%)")
        
        # Save significant only
        med_sig = med_combined[med_combined['significant']]
        med_sig_output = output_dir / 'medication_group_comparisons_significant.csv'
        med_sig.to_csv(med_sig_output, index=False)
        logging.info(f"   Significant only: {med_sig_output}")
    
    if all_ame:
        ame_combined = pd.concat(all_ame, ignore_index=True)
        ame_output = output_dir / 'average_marginal_effects.csv'
        ame_combined.to_csv(ame_output, index=False)
        logging.info(f"\n✅ Average Marginal Effects saved: {ame_output}")
        logging.info(f"   Total comparisons: {len(ame_combined)}")
        logging.info(f"   Significant: {ame_combined['significant'].sum()} ({ame_combined['significant'].mean()*100:.1f}%)")
        
        # Save significant only
        ame_sig = ame_combined[ame_combined['significant']]
        ame_sig_output = output_dir / 'average_marginal_effects_significant.csv'
        ame_sig.to_csv(ame_sig_output, index=False)
        logging.info(f"   Significant only: {ame_sig_output}")
    
    if all_logistic_regression:
        lr_combined = pd.concat(all_logistic_regression, ignore_index=True)
        
        logging.info(f"\n✅ Logistic Regression Analysis Complete")
        logging.info(f"   Total coefficients: {len(lr_combined)}")
        logging.info(f"   Outcomes analyzed: {lr_combined['outcome'].nunique()}")
        
        # Create formatted coefficient tables (pivot tables)
        create_formatted_logistic_regression_tables(all_logistic_regression, output_dir, models)
    
    logging.info("\n" + "=" * 80)
    logging.info("✅ ENHANCED ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"\nAll results saved to: {output_dir}/")
    logging.info("\nFiles generated:")
    logging.info("  1. odds_ratios_all.csv - Full odds ratio table (group vs reference)")
    logging.info("  2. odds_ratios_significant.csv - Significant odds ratios only")
    logging.info("  3. medication_group_comparisons_all.csv - Medication patterns (group vs group)")
    logging.info("  4. medication_group_comparisons_significant.csv - Significant medication differences")
    logging.info("  5. average_marginal_effects.csv - AME for all comparisons")
    logging.info("  6. average_marginal_effects_significant.csv - Significant AMEs only")
    logging.info("  7. logistic_regression_coef_[outcome]_[class].csv - Coefficient tables (one per outcome class)")
    logging.info("\nNOTE: Group-to-group comparisons (not baseline) are used for medications")
    logging.info("      as recommended for more meaningful statistical comparisons.")
    logging.info("\nLOGISTIC REGRESSION REFERENCE CATEGORIES (DROPPED FROM TABLES):")
    logging.info("  - White (race), cisgender man (gender), heterosexual (orientation)")
    logging.info("  - upper class (SES), white collar (occupation), English proficient (language)")
    logging.info("  - urban (geography), young (age)")


if __name__ == "__main__":
    main()

