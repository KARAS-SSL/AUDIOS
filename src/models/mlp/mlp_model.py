import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A modular Multi-Layer Perceptron (MLP) class.

    Args
    ----
    input_dim : int
        Dimension of input features.
    hidden_dim : int
        Dimension of the hidden layer.
    output_dim : int
        Number of output classes.
    dropout_prob : float
        Probability of dropout for regularization.
    activation : str
        Activation function to use ('ReLU', 'Tanh', 'Sigmoid', etc.).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.5, activation: str = "ReLU"):
        super(MLP, self).__init__()

        # Dynamically assign the activation function
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU(),
        }
        assert activation in activations, f"Unsupported activation: {activation}"
        self.activation = activations[activation]

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout for regularization
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines how input data flows through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
