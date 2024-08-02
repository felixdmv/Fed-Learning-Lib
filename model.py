import torch
import torch.nn as nn

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, dropout_rate):
        super(SimpleBinaryClassifier, self).__init__()
        self.hidden_dim = hidden_dim        
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Convertir la salida a probabilidad
        return x

def create_model(input_size, hidden_dim, num_layers, output_size, device):
    """Create a new SimpleBinaryClassifier model instance."""
    return SimpleBinaryClassifier(input_size, hidden_dim, num_layers, output_size, dropout_rate=0.5).to(device)
