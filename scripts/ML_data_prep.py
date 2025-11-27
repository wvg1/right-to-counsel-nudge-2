import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### load data ###
df = pd.read_excel("data/baseline_data_11_26_25.xlsx")

### select features ###
feature_columns = [
    'plaintiff_rep',                  #binary: plaintiff has attorney
    'amount_owed',                    #numeric: dollars owed at filing
    'defendant_appearance',           #binary: defendant appeared before first hearing
    'hearing_held',                   #binary: hearing occurred
    'defendant_hearing_attendance',   #binary: defendant at hearing
    'tenant_rep_merged',              #binary: tenant has representation
]

#target variables (predict separately)
target_columns = [
    'court_displacement',             #binary: court documents indicate the tenant moved
    'writ_final',                     #binary: writ was issued and ultimately not vacated
    'old_final',                      #binary: record protection order was issued and ultimately not vacated
]

#extract features and targets
X = df[feature_columns].copy()
y_court_disp = df['court_displacement'].copy()
y_writ = df['writ_final'].copy()
y_old = df['old_final'].copy()

print(f"feature matrix shape: {X.shape}")
print(f"court displacement target: {y_court_disp.shape}")
print(f"writ final target: {y_writ.shape}")
print(f"OLD final target: {y_old.shape}\n")

### clean data ###
#drop rows with any missing values
valid_idx = X.notna().all(axis=1) & y_court_disp.notna() & y_writ.notna() & y_old.notna()
X = X[valid_idx]
y_court_disp = y_court_disp[valid_idx]
y_writ = y_writ[valid_idx]
y_old = y_old[valid_idx]

print(f"After removing missing values: {X.shape[0]} cases\n")

#convert boolean columns to float
X = X.astype(float)
y_court_disp = y_court_disp.astype(float)
y_writ = y_writ.astype(float)
y_old = y_old.astype(float)

### standardize features ###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"feature statistics after standardization:")
print(f"  mean: {X_scaled.mean(axis=0).round(3)}")
print(f"  std:  {X_scaled.std(axis=0).round(3)}\n")

### train/test split ###

#80/20 split
X_train, X_test, y_train_court, y_test_court, y_train_writ, y_test_writ, y_train_old, y_test_old = train_test_split(
    X_scaled, y_court_disp, y_writ, y_old, test_size=0.2, random_state=42
)

print(f"train set: {X_train.shape[0]} cases")
print(f"test set:  {X_test.shape[0]} cases\n")

### convert to pytorch tensors ###
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()

y_train_court_tensor = torch.from_numpy(y_train_court.values).float().unsqueeze(1)
y_test_court_tensor = torch.from_numpy(y_test_court.values).float().unsqueeze(1)

y_train_writ_tensor = torch.from_numpy(y_train_writ.values).float().unsqueeze(1)
y_test_writ_tensor = torch.from_numpy(y_test_writ.values).float().unsqueeze(1)

y_train_old_tensor = torch.from_numpy(y_train_old.values).float().unsqueeze(1)
y_test_old_tensor = torch.from_numpy(y_test_old.values).float().unsqueeze(1)

print(f"tensor shapes:")
print(f"  X_train: {X_train_tensor.shape}  (cases, features)")
print(f"  y_train (court_displacement): {y_train_court_tensor.shape}  (cases, 1)")
print(f"  X_test:  {X_test_tensor.shape}   (cases, features)")
print(f"  y_test (court_displacement):  {y_test_court_tensor.shape}   (cases, 1)\n")

### check outcome distribution ###
print(f"outcome distribution in training set:")
print(f"  court displacement: {int(y_train_court_tensor.sum().item())}/{len(y_train_court_tensor)} ({100*y_train_court_tensor.mean().item():.1f}%)")
print(f"  writ final: {int(y_train_writ_tensor.sum().item())}/{len(y_train_writ_tensor)} ({100*y_train_writ_tensor.mean().item():.1f}%)")
print(f"  old final: {int(y_train_old_tensor.sum().item())}/{len(y_train_old_tensor)} ({100*y_train_old_tensor.mean().item():.1f}%)\n")

print("=== data preparation complete ===")