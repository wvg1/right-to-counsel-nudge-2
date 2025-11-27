import torch
import torch.nn as nn

### define nn architecture ###

class EvictionNet(nn.Module):
    """
    Neural network to predict eviction case outcomes.
    
    Architecture:
    - Input: 6 features (procedural variables)
    - Hidden layer 1: 16 neurons + ReLU activation
    - Hidden layer 2: 8 neurons + ReLU activation
    - Output: 1 neuron + Sigmoid (binary classification)
    """
    
    def __init__(self, input_size=6, hidden_size_1=16, hidden_size_2=8):
        super(EvictionNet, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(input_size, hidden_size_1)        # input → hidden 1
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)     # hidden 1 → hidden 2
        self.fc3 = nn.Linear(hidden_size_2, 1)                 # hidden 2 → output
        
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

# create 3 separate models, one for each outcome
model_court_displacement = EvictionNet()
model_writ_final = EvictionNet()
model_old_final = EvictionNet()

print("=== neural network models created ===\n")

# show architecture
print("model architecture:")
print(model_court_displacement)
print()

# count parameters
total_params = sum(p.numel() for p in model_court_displacement.parameters())
trainable_params = sum(p.numel() for p in model_court_displacement.parameters() if p.requires_grad)

print(f"total parameters: {total_params}")
print(f"trainable parameters: {trainable_params}\n")

# test forward pass with dummy data
print("testing forward pass with dummy data:")
dummy_input = torch.randn(5, 6)  # batch of 5 samples, 6 features
dummy_output = model_court_displacement(dummy_input)

print(f"input shape: {dummy_input.shape}  (batch_size=5, features=6)")
print(f"output shape: {dummy_output.shape}  (batch_size=5, predictions=1)")
print(f"sample predictions (probabilities): {dummy_output.squeeze()}")
