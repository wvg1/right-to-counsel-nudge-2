import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

### define nn architecture ###

class EvictionNet(nn.Module):
    """
    Neural network to predict eviction case outcomes.
    
    Architecture:
    - Input: 4 features (filing-date characteristics)
    - Hidden layer 1: 16 neurons + ReLU activation
    - Hidden layer 2: 8 neurons + ReLU activation
    - Output: 1 neuron + Sigmoid (binary classification)
    """
    
    def __init__(self, input_size=4, hidden_size_1=16, hidden_size_2=8):
        super(EvictionNet, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)
        
        # activation functions
        self.relu = nn.ReLU()                                   # nonlinear activation
        self.sigmoid = nn.Sigmoid()                             # convert to probability
    
    def forward(self, x):
        """
        Forward pass: data flows through network.
        
        Args:
            x: input tensor of shape (batch_size, input_size)
        
        Returns:
            output: probability tensor of shape (batch_size, 1)
        """
        
        # hidden layer 1
        x = self.fc1(x)                 # linear transformation
        x = self.relu(x)                # apply nonlinearity
        
        # hidden layer 2
        x = self.fc2(x)                 # linear transformation
        x = self.relu(x)                # apply nonlinearity
        
        # output layer
        x = self.fc3(x)                 # linear transformation
        x = self.sigmoid(x)             # convert to probability (0-1)
        
        return x

### instantiate models ###

target_columns = [
    'defendant_appearance',
    'hearing_held',
    'defendant_hearing_attendance',
    'tenant_rep_merged',
    'writ_final',
    'dismissal_final',
    'old_final',
    'court_displacement',
]

models = {col: EvictionNet() for col in target_columns}

print("=== neural network models created ===\n")

# show architecture
print("model architecture:")
print(models['defendant_appearance'])
print()

# count parameters
total_params = sum(p.numel() for p in models['defendant_appearance'].parameters())
trainable_params = sum(p.numel() for p in models['defendant_appearance'].parameters() if p.requires_grad)

print(f"total parameters per model: {total_params}")
print(f"trainable parameters per model: {trainable_params}")
print(f"total parameters across {len(models)} models: {total_params * len(models)}\n")

# test forward pass with dummy data
print("testing forward pass with dummy data:")
dummy_input = torch.randn(5, 4)  # batch of 5 samples, 4 features
dummy_output = models['defendant_appearance'](dummy_input)

print(f"input shape: {dummy_input.shape}  (batch_size=5, features=4)")
print(f"output shape: {dummy_output.shape}  (batch_size=5, predictions=1)")
print(f"sample predictions (probabilities): {dummy_output.squeeze()}\n")

### create dataloaders ###

# assuming tensors are loaded from data prep script
batch_size = 32

# training dataloader - stack all targets
train_dataset = TensorDataset(
    X_train_tensor,
    *[y_train_tensors[col] for col in target_columns]
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# validation dataloader
val_dataset = TensorDataset(
    X_val_tensor,
    *[y_val_tensors[col] for col in target_columns]
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# test dataloader
test_dataset = TensorDataset(
    X_test_tensor,
    *[y_test_tensors[col] for col in target_columns]
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"dataloaders created:")
print(f"  train batches: {len(train_loader)} (batch_size={batch_size})")
print(f"  val batches:   {len(val_loader)} (batch_size={batch_size})")
print(f"  test batches:  {len(test_loader)} (batch_size={batch_size})\n")

### define training utilities ###

criterion = nn.BCELoss()  # binary cross-entropy for binary classification

optimizers = {col: torch.optim.Adam(models[col].parameters(), lr=0.001) for col in target_columns}

print(f"predicting {len(target_columns)} outcomes: {', '.join(target_columns)}")