import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, dropout_rate):
        """
        Initializes the SimpleBinaryClassifier model.

        Args:
            input_size (int): The size of the input features.
            hidden_dim (int): The dimension of the hidden layer.
            num_layers (int): The number of layers in the model.
            output_size (int): The size of the output.
            dropout_rate (float): The dropout rate for regularization.

        Returns:
            None
        """
        super(SimpleBinaryClassifier, self).__init__()
        self.hidden_dim = hidden_dim        
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


def create_model(input_size, hidden_dim, num_layers, output_size, device):
    """Create a new SimpleBinaryClassifier model instance."""
    return SimpleBinaryClassifier(input_size, hidden_dim, num_layers, output_size, dropout_rate=0.5).to(device)
