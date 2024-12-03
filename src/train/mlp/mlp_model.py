import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5, activation='ReLU'):
        """
        A modular Multi-Layer Perceptron (MLP) class.
        
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Number of output classes.
            activation (str): Activation function to use ('ReLU', 'Tanh', 'Sigmoid', etc.).
            dropout_prob (float): Probability of dropout for regularization.
        """
        super(MLP, self).__init__()
        
        # Dynamically assign the activation function
        activations = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU()
        }
        assert activation in activations, f"Unsupported activation: {activation}"
        self.activation = activations[activation]
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout for regularization 
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x) 
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
