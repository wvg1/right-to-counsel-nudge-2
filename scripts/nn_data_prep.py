import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### load data ###
df = pd.read_excel("data/baseline_data_11_26_25.xlsx")

print(f"initial dataset: {len(df)} cases\n")

### filter to Pierce County only (valid CBGs start with 53053) ###
df = df[df['census_block_group'].astype(str).str.startswith('53053')].copy()
n_removed_cbg = len(df) - len(df)
print(f"After filtering to Pierce County CBGs (53053): {len(df)} cases\n")

### filter to zips with Pierce County CBGs ###
pierce_zips = df[df['census_block_group'].astype(str).str.startswith('53053')]['address_zip'].dropna().unique()
df = df[df['address_zip'].isin(pierce_zips)].copy()

print(f"Valid Pierce County zips: {len(pierce_zips)}")
print(f"Cases with valid zips: {len(df)}\n")

### select features ###
feature_columns = [ 
    'amount_owed',                    #numeric: dollars owed at filing
    'address_zip',                    #categorical: geographic identifier
    'plaintiff_rep',                  #binary: plaintiff has attorney
]

# target variables (predict separately)
target_columns = [
    'defendant_appearance',           #binary: defendant appeared before first hearing
    'hearing_held',                   #binary: hearing occurred
    'defendant_hearing_attendance',   #binary: defendant at hearing
    'defendant_rep_merged',              #binary: tenant has representation
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

# one-hot encode address_zip
X = pd.get_dummies(X, columns=['address_zip'], drop_first=True, dtype=float)

# convert other columns to float
X['amount_owed'] = X['amount_owed'].astype(float)
X['plaintiff_rep'] = X['plaintiff_rep'].astype(float)

for col in target_columns:
    y_dict[col] = y_dict[col].astype(float)

print(f"feature matrix shape after one-hot encoding: {X.shape}")
print(f"  {X.shape[1]} features total\n")

### standardize numeric features only ###
# identify which columns are one-hot encoded zips (they stay 0/1)
numeric_cols = ['amount_owed', 'plaintiff_rep']
zip_cols = [col for col in X.columns if col.startswith('address_zip_')]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"feature statistics after standardization (numeric features):")
print(f"  amount_owed: mean={X['amount_owed'].mean():.3f}, std={X['amount_owed'].std():.3f}")
print(f"  plaintiff_rep: mean={X['plaintiff_rep'].mean():.3f}, std={X['plaintiff_rep'].std():.3f}\n")

### train/test/validate split (60/20/20) ###

# convert to numpy for splitting
X_np = X.values

# first split: 80/20 (train+val / test)
split_data = train_test_split(
    X_np, 
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

print(f"tensor shapes:")
print(f"  X_train: {X_train_tensor.shape}  (cases, features)")
print(f"  X_val:   {X_val_tensor.shape}   (cases, features)")
print(f"  X_test:  {X_test_tensor.shape}   (cases, features)\n")

### check outcome distribution ###
for col in target_columns:
    print(f"{col}:")
    print(f"  train: {int(y_train_tensors[col].sum().item())}/{len(y_train_tensors[col])} ({100*y_train_tensors[col].mean().item():.1f}%)")
    print(f"  val:   {int(y_val_tensors[col].sum().item())}/{len(y_val_tensors[col])} ({100*y_val_tensors[col].mean().item():.1f}%)")
    print(f"  test:  {int(y_test_tensors[col].sum().item())}/{len(y_test_tensors[col])} ({100*y_test_tensors[col].mean().item():.1f}%)")
    print()

### save tensors for model training ###
torch.save({
    'X_train': X_train_tensor,
    'X_val': X_val_tensor,
    'X_test': X_test_tensor,
    'y_train': y_train_tensors,
    'y_val': y_val_tensors,
    'y_test': y_test_tensors,
    'target_columns': target_columns,
}, 'data/tensors.pt')

print(f"tensors saved to 'data/tensors.pt'\n")