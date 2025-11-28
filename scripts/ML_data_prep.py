import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### load data ###
df = pd.read_excel("data/baseline_data_11_26_25.xlsx")

### select features (filing-date only) ###
feature_columns = [
    'pocket_service',                 #binary: served via pocket service
    'amount_owed',                    #numeric: dollars owed at filing
    'CBG',                            #numeric: census block group
    'plaintiff_rep',                  #binary: plaintiff has attorney
]

# target variables (predict separately)
target_columns = [
    'defendant_appearance',           #binary: defendant appeared before first hearing
    'hearing_held',                   #binary: hearing occurred
    'defendant_hearing_attendance',   #binary: defendant at hearing
    'tenant_rep_merged',              #binary: tenant has representation
    'writ_final',                     #binary: writ was issued and ultimately not vacated
    'dismissal_final',                #binary: case was dismissed and ultimately not reinstated
    'old_final',                      #binary: record protection order was issued and ultimately not vacated
    'court_displacement',             #binary: court documents indicate the tenant moved
]

# extract features and targets
X = df[feature_columns].copy()
y_dict = {col: df[col].copy() for col in target_columns}

print(f"feature matrix shape: {X.shape}")
for col in target_columns:
    print(f"{col}: {y_dict[col].shape}")
print()

### clean data ###
# drop rows with any missing values in features
valid_idx = X.notna().all(axis=1)

# also require valid targets
for col in target_columns:
    valid_idx = valid_idx & y_dict[col].notna()

X = X[valid_idx]
for col in target_columns:
    y_dict[col] = y_dict[col][valid_idx]

print(f"After removing missing values: {X.shape[0]} cases\n")

# convert to float
X = X.astype(float)
for col in target_columns:
    y_dict[col] = y_dict[col].astype(float)

### standardize features ###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"feature statistics after standardization:")
print(f"  mean: {X_scaled.mean(axis=0).round(3)}")
print(f"  std:  {X_scaled.std(axis=0).round(3)}\n")

### train/test/validate split (60/20/20) ###

# first split: 80/20 (train+val / test)
split_data = train_test_split(
    X_scaled, 
    *[y_dict[col] for col in target_columns],
    test_size=0.2, 
    random_state=42
)

X_temp = split_data[0]
X_test = split_data[1]
y_temp_dict = {}
y_test_dict = {}

for i, col in enumerate(target_columns):
    y_temp_dict[col] = split_data[2 + i*2]
    y_test_dict[col] = split_data[2 + i*2 + 1]

# second split: 75/25 of temp (60/20 of original)
split_data2 = train_test_split(
    X_temp,
    *[y_temp_dict[col] for col in target_columns],
    test_size=0.25,
    random_state=42
)

X_train = split_data2[0]
X_val = split_data2[1]
y_train_dict = {}
y_val_dict = {}

for i, col in enumerate(target_columns):
    y_train_dict[col] = split_data2[2 + i*2]
    y_val_dict[col] = split_data2[2 + i*2 + 1]

print(f"train set: {X_train.shape[0]} cases (60%)")
print(f"val set:   {X_val.shape[0]} cases (20%)")
print(f"test set:  {X_test.shape[0]} cases (20%)\n")

### convert to pytorch tensors ###
X_train_tensor = torch.from_numpy(X_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
X_test_tensor = torch.from_numpy(X_test).float()

y_train_tensors = {}
y_val_tensors = {}
y_test_tensors = {}

for col in target_columns:
    y_train_tensors[col] = torch.from_numpy(y_train_dict[col].values).float().unsqueeze(1)
    y_val_tensors[col] = torch.from_numpy(y_val_dict[col].values).float().unsqueeze(1)
    y_test_tensors[col] = torch.from_numpy(y_test_dict[col].values).float().unsqueeze(1)

# tensor shapes
print(f"  X_train: {X_train_tensor.shape}  (cases, features)")
print(f"  X_val:   {X_val_tensor.shape}   (cases, features)")
print(f"  X_test:  {X_test_tensor.shape}   (cases, features)\n")

### check outcome distribution ###
for col in target_columns:
    print(f"\n{col}:")
    print(f"  train: {int(y_train_tensors[col].sum().item())}/{len(y_train_tensors[col])} ({100*y_train_tensors[col].mean().item():.1f}%)")
    print(f"  val:   {int(y_val_tensors[col].sum().item())}/{len(y_val_tensors[col])} ({100*y_val_tensors[col].mean().item():.1f}%)")
    print(f"  test:  {int(y_test_tensors[col].sum().item())}/{len(y_test_tensors[col])} ({100*y_test_tensors[col].mean().item():.1f}%)")
