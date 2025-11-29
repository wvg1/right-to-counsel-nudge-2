import pandas as pd
import numpy as np
from causalml.inference.tree import CausalTreeRegressor
from causalml.inference.forest import CausalForestDML
from sklearn.preprocessing import StandardScaler
import json
import os

### load data ###

df = pd.read_excel("data/baseline_data_11_26_25.xlsx")

# filter to Pierce County
df = df[df['census_block_group'].astype(str).str.startswith('53053')].copy()
pierce_zips = df[df['census_block_group'].astype(str).str.startswith('53053')]['address_zip'].dropna().unique()
df = df[df['address_zip'].isin(pierce_zips)].copy()

# target variables
target_columns = [
    'defendant_appearance',
    'hearing_held',
    'defendant_hearing_attendance',
    'defendant_rep_merged',
    'writ_final',
    'dismissal_final',
    'old_final',
    'court_displacement',
]

# treatment column
treatment_col = 'treat'

# drop rows with missing values
valid_idx = df[target_columns + [treatment_col]].notna().all(axis=1)
df = df[valid_idx].copy()

print(f"analyzing {len(df)} cases")
print(f"treatment assignment: {df[treatment_col].sum()} treated, {(1-df[treatment_col]).sum()} control\n")

### prepare features ###

feature_columns = [ 
    'amount_owed',
    'address_zip',
    'plaintiff_rep',
]

X = df[feature_columns].copy()
X = pd.get_dummies(X, columns=['address_zip'], drop_first=True, dtype=float)

# standardize numeric features
numeric_cols = ['amount_owed', 'plaintiff_rep']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# convert to float
X = X.astype(float)

# treatment and Stage 1 outcomes
W = df[treatment_col].values.astype(float)
stage1_outcomes = ['defendant_appearance', 'defendant_hearing_attendance', 'defendant_rep_merged']

print(f"feature matrix shape: {X.shape}")
print(f"treatment vector shape: {W.shape}\n")

### create results directory ###

os.makedirs("results/causal_forest", exist_ok=True)

### fit causal forests for Stage 1 outcomes ###

cate_results = {}

for outcome in stage1_outcomes:
    print(f"Outcome: {outcome}")
    
    Y = df[outcome].values.astype(float)
    
    # fit causal forest using DML (double machine learning)
    cf = CausalForestDML(
        model_y=None,  # uses default model
        model_t=None,  # uses default model
        n_trees=100,
        random_state=42,
        n_jobs=-1
    )
    
    cf.fit(X.values, W, Y)
    
    # get CATE (conditional average treatment effect) for each unit
    cate = cf.predict(X.values)
    
    # get ATE (average treatment effect)
    ate = cate.mean()
    
    # calculate treatment effect by treatment/control
    treated_effect = cate[W == 1].mean()
    control_effect = cate[W == 0].mean()
    
    # get feature importance
    feature_importance = cf.feature_importances_
    
    # sort features by importance
    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"  ATE (Average Treatment Effect): {ate:.4f}")
    print(f"  Among treated: {treated_effect:.4f}")
    print(f"  Among control: {control_effect:.4f}")
    print(f"  Effect size (treated - control): {treated_effect - control_effect:.4f}\n")
    
    print(f"  Top 5 features by importance:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")
    print()
    
    # store results
    cate_results[outcome] = {
        'ate': float(ate),
        'treated_effect': float(treated_effect),
        'control_effect': float(control_effect),
        'feature_importance': importance_df.to_dict('records'),
        'cate': cate.tolist(),  # individual treatment effects
    }

# save results
with open('results/causal_forest/stage1_treatment_effects.json', 'w') as f:
    # convert lists to summaries for JSON serialization
    results_to_save = {}
    for outcome, result in cate_results.items():
        results_to_save[outcome] = {
            'ate': result['ate'],
            'treated_effect': result['treated_effect'],
            'control_effect': result['control_effect'],
            'cate_mean': np.mean(result['cate']),
            'cate_std': np.std(result['cate']),
            'feature_importance': result['feature_importance'][:10],  # top 10 features
        }
    json.dump(results_to_save, f, indent=2)

print(f"{'='*70}")
print("Stage 1 treatment effects saved to 'results/causal_forest/stage1_treatment_effects.json'")
print(f"{'='*70}\n")

### heterogeneous treatment effects ###

print(f"{'='*70}")
print("HETEROGENEOUS TREATMENT EFFECTS: By Amount Owed Quartiles")
print(f"{'='*70}\n")

# look at treatment effects by amount owed quartile
df_analysis = df[['amount_owed', treatment_col] + stage1_outcomes].copy()

for outcome in stage1_outcomes:
    print(f"\n{outcome}:")
    df_analysis['quartile'] = pd.qcut(df_analysis['amount_owed'], q=4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    
    for quartile in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        subset = df_analysis[df_analysis['quartile'] == quartile]
        treated = subset[subset[treatment_col] == 1][outcome].mean()
        control = subset[subset[treatment_col] == 0][outcome].mean()
        effect = treated - control
        
        print(f"  {quartile:15s}: treated={treated:.3f}, control={control:.3f}, effect={effect:+.3f}")
