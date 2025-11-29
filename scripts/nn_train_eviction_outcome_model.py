import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import json
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

### load tensors ###

checkpoint = torch.load('data/tensors.pt')
X_train_tensor = checkpoint['X_train']
X_val_tensor = checkpoint['X_val']
X_test_tensor = checkpoint['X_test']
y_train_tensors = checkpoint['y_train']
y_val_tensors = checkpoint['y_val']
y_test_tensors = checkpoint['y_test']
target_columns = checkpoint['target_columns']

print(f"tensors loaded from 'data/tensors.pt'\n")

### Stage 1 outcomes (actual values from data) ###

stage1_outcomes = [
    'defendant_appearance',
    'defendant_hearing_attendance',
    'defendant_rep_merged',
]

stage2_targets = [
    'writ_final',
    'dismissal_final',
    'old_final',
    'court_displacement',
]

print(f"Stage 1 outcomes (actual values): {stage1_outcomes}")
print(f"Stage 2 targets: {stage2_targets}\n")

### define nn architecture ###

class EvictionNet(nn.Module):
    """Neural network to predict eviction case outcomes."""
    
    def __init__(self, input_size=85, hidden_size_1=32, hidden_size_2=16):
        super(EvictionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

### create directories ###

os.makedirs("models/stage2", exist_ok=True)
os.makedirs("results/stage2", exist_ok=True)

### hyperparameter configuration ###

config = {
    'num_epochs': 100,
    'patience': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hidden_size_1': 32,
    'hidden_size_2': 16,
}

print(f"training configuration:")
print(json.dumps(config, indent=2))
print()

### device ###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}\n")

### training and validation functions ###

def train_epoch(model, optimizer, train_loader, criterion, device, target_idx):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in train_loader:
        X_batch = batch_data[0].to(device)
        y_batch = batch_data[target_idx + 1].to(device)
        
        if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
            continue
        
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        if np.isnan(loss.item()):
            continue
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

def validate(model, val_loader, criterion, device, target_idx):
    """Evaluate model on validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            X_batch = batch_data[0].to(device)
            y_batch = batch_data[target_idx + 1].to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def test_model(model, test_loader, criterion, device, target_idx):
    """Evaluate model on test set and return loss and predictions."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            X_batch = batch_data[0].to(device)
            y_batch = batch_data[target_idx + 1].to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, predictions, targets

### CREATE STAGE 2 TENSORS (combining actual Stage 1 outcomes with original features) ###

print(f"{'='*70}")
print("Creating Stage 2 Tensors (original features + actual Stage 1 outcomes)")
print(f"{'='*70}\n")

# concatenate actual Stage 1 outcomes with original features
X_train_s2 = np.hstack([
    X_train_tensor.numpy(),
    np.hstack([y_train_tensors[col].numpy() for col in stage1_outcomes])
])

X_val_s2 = np.hstack([
    X_val_tensor.numpy(),
    np.hstack([y_val_tensors[col].numpy() for col in stage1_outcomes])
])

X_test_s2 = np.hstack([
    X_test_tensor.numpy(),
    np.hstack([y_test_tensors[col].numpy() for col in stage1_outcomes])
])

X_train_s2_tensor = torch.from_numpy(X_train_s2).float()
X_val_s2_tensor = torch.from_numpy(X_val_s2).float()
X_test_s2_tensor = torch.from_numpy(X_test_s2).float()

print(f"Stage 2 feature dimensions:")
print(f"  Original features: 85")
print(f"  Stage 1 outcomes: {len(stage1_outcomes)} (appearance, hearing_attendance, rep_merged)")
print(f"  Total Stage 2 features: {X_train_s2.shape[1]}\n")

print(f"Stage 2 tensor shapes:")
print(f"  X_train: {X_train_s2_tensor.shape}")
print(f"  X_val:   {X_val_s2_tensor.shape}")
print(f"  X_test:  {X_test_s2_tensor.shape}\n")

### STAGE 2: Train Case Outcome Models ###

print(f"{'='*70}")
print("STAGE 2: Training Case Outcome Models")
print(f"{'='*70}\n")

# create Stage 2 dataloaders
train_dataset_s2 = TensorDataset(
    X_train_s2_tensor,
    *[y_train_tensors[col] for col in stage2_targets]
)
train_loader_s2 = DataLoader(train_dataset_s2, batch_size=config['batch_size'], shuffle=True)

val_dataset_s2 = TensorDataset(
    X_val_s2_tensor,
    *[y_val_tensors[col] for col in stage2_targets]
)
val_loader_s2 = DataLoader(val_dataset_s2, batch_size=config['batch_size'], shuffle=False)

test_dataset_s2 = TensorDataset(
    X_test_s2_tensor,
    *[y_test_tensors[col] for col in stage2_targets]
)
test_loader_s2 = DataLoader(test_dataset_s2, batch_size=config['batch_size'], shuffle=False)

# instantiate Stage 2 models
input_size_s2 = X_train_s2.shape[1]
models_s2 = {col: EvictionNet(input_size=input_size_s2).to(device) for col in stage2_targets}
optimizers_s2 = {col: torch.optim.Adam(models_s2[col].parameters(), lr=config['learning_rate']) for col in stage2_targets}

criterion = nn.BCELoss()
training_history_s2 = {col: {'train_loss': [], 'val_loss': [], 'epochs': 0, 'training_time': 0} for col in stage2_targets}
model_info_s2 = {}

# train Stage 2 models
for target_idx, target_col in enumerate(stage2_targets):
    model = models_s2[target_col]
    optimizer = optimizers_s2[target_col]
    
    print(f"training: {target_col}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, optimizer, train_loader_s2, criterion, device, target_idx)
        val_loss = validate(model, val_loader_s2, criterion, device, target_idx)
        
        training_history_s2[target_col]['train_loss'].append(train_loss)
        training_history_s2[target_col]['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  |  train_loss: {train_loss:.4f}  |  val_loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"models/stage2/{target_col}_best.pt")
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, f"models/stage2/{target_col}_checkpoint.pt")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"  early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.4f})")
                break
    
    elapsed_time = time.time() - start_time
    training_history_s2[target_col]['epochs'] = best_epoch
    training_history_s2[target_col]['training_time'] = elapsed_time
    model_info_s2[target_col] = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'training_time_seconds': elapsed_time,
    }
    
    print(f"  complete. best val_loss: {best_val_loss:.4f} at epoch {best_epoch} ({elapsed_time:.2f}s)\n")

# save Stage 2 history
with open('results/stage2/training_history.pkl', 'wb') as f:
    pickle.dump(training_history_s2, f)
with open('results/stage2/model_info.json', 'w') as f:
    json.dump(model_info_s2, f, indent=2)

# plot Stage 2 loss curves
print("generating Stage 2 loss curves...")
for col in stage2_targets:
    plt.figure(figsize=(10, 6))
    plt.plot(training_history_s2[col]['train_loss'], label='train loss', linewidth=2)
    plt.plot(training_history_s2[col]['val_loss'], label='val loss', linewidth=2)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('loss (BCE)', fontsize=12)
    plt.title(f'Stage 2: {col} - Training History', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/stage2/{col}_loss_curve.png', dpi=150)
    plt.close()

print("Stage 2 loss curves saved to 'results/stage2/'\n")

### STAGE 2: Evaluate ###

print(f"{'='*70}")
print("STAGE 2: Evaluating Case Outcome Models")
print(f"{'='*70}\n")

performance_metrics_s2 = {}

for target_idx, target_col in enumerate(stage2_targets):
    model = models_s2[target_col]
    model.load_state_dict(torch.load(f"models/stage2/{target_col}_best.pt"))
    
    test_loss, test_preds, test_targets = test_model(model, test_loader_s2, criterion, device, target_idx)
    
    auc = roc_auc_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds > 0.5, average='binary')
    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds > 0.5).ravel()
    specificity = tn / (tn + fp)
    
    performance_metrics_s2[target_col] = {
        'test_loss': test_loss,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
    }
    
    print(f"{target_col:30s} | AUC: {auc:.4f} | F1: {f1:.4f}")

with open('results/stage2/performance_metrics.json', 'w') as f:
    json.dump(performance_metrics_s2, f, indent=2)