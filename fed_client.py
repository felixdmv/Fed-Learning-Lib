import os
import warnings
import numpy as np
from collections import OrderedDict
from flwr.client import NumPyClient, start_client
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from argparse import ArgumentParser
import yaml
from typing import Dict, Any, List
from model import create_model

# Load configuration from YAML file
def load_config(yaml_file: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config('configuracion.yaml')

# Read configurations from YAML
save_model_directory = config['model']['save_model_directory']
input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
output_dim = config['model']['output_dim']

learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
dropout_rate = config['training']['dropout_rate']

num_rounds = config['server']['num_rounds']
server_address = config['server']['server_address']

num_clients = config['client']['num_clients']

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(save_model_directory), exist_ok=True)

# Argument parser to allow default values
parser = ArgumentParser(description="Flower Client")
parser.add_argument('--server_address', type=str, default=server_address, help="Address of the Flower server")
parser.add_argument('--partition_id', type=int, default=0, help="Partition ID for the client")
parser.add_argument('--round', type=int, default=1, help="Current round number")
args = parser.parse_args()

# Load and preprocess data
df = pd.read_csv('processed_data.csv')
X = df.drop(columns=['diabetesMed_Yes']).values
y = df['diabetesMed_Yes'].values
y = (y > 0).astype(int)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into partitions
partition_size = len(X) // num_clients
partition_start = args.partition_id * partition_size
partition_end = partition_start + partition_size if args.partition_id != num_clients - 1 else len(X)

X_train, X_test, y_train, y_test = train_test_split(X[partition_start:partition_end], y[partition_start:partition_end], test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

trainset = CustomDataset(X_train, y_train)
testset = CustomDataset(X_test, y_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Client(NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
    

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        for epoch in range(epochs):
            for batch in self.trainloader:
                X_batch, y_batch = batch
                y_batch = y_batch.unsqueeze(1)  # Asegúrate de que el objetivo tenga tamaño [batch_size, 1]

                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.float())
                loss.backward()
                self.optimizer.step()
        
        # Exportar el modelo a TorchScript y guardarlo
        datetime = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        model_scripted = torch.jit.script(model)
        model_file_path = os.path.join(save_model_directory, f"client_{args.partition_id}_{datetime}.pth")
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        torch.save(model.state_dict(), model_file_path)
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        model_scripted.save(model_file_path)
        print(f"Model saved to {model_file_path}")

        # # Save model after training        
        # os.makedirs(os.path.dirname(save_model_directory), exist_ok=True)
        # model_path = os.path.join(save_model_directory, f'client_{args.partition_id}_{args.round:08d}.pth')
        # torch.save(self.model.state_dict(), model_path)
        
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.testloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_batch = y_batch.unsqueeze(1)
                outputs = self.model(X_batch)
                loss += self.criterion(outputs, y_batch.float()).item()
                predicted = (outputs > 0.5).int()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        loss /= len(self.testloader.dataset)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "loss": loss
        }
        
        return float(loss), len(self.testloader.dataset), metrics

model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)
client = Client(model, trainloader, testloader)

if __name__ == "__main__":
    start_client(server_address=args.server_address, client=client.to_client())


# import os
# import warnings
# from collections import OrderedDict
# from flwr.client import NumPyClient, start_client
# import torch
# from torch.optim import Adam
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, f1_score
# import pandas as pd
# from argparse import ArgumentParser
# import yaml
# from typing import Dict, Any
# from model import create_model

# # Load configuration from YAML file
# def load_config(yaml_file: str) -> Dict[str, Any]:
#     """Load configuration from a YAML file."""
#     with open(yaml_file, 'r') as file:
#         config = yaml.safe_load(file)
#     return config

# config = load_config('configuracion.yaml')

# # Read configurations from YAML
# save_model_directory = config['model']['save_model_directory']
# input_dim = config['model']['input_dim']
# hidden_dim = config['model']['hidden_dim']
# num_layers = config['model']['num_layers']
# output_dim = config['model']['output_dim']

# learning_rate = config['training']['learning_rate']
# batch_size = config['training']['batch_size']
# epochs = config['training']['epochs']
# dropout_rate = config['training']['dropout_rate']

# num_rounds = config['server']['num_rounds']
# server_address = config['server']['server_address']

# num_clients = config['client']['num_clients']

# warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.makedirs(os.path.dirname(save_model_directory), exist_ok=True)

# # Argument parser to allow default values
# parser = ArgumentParser(description="Flower Client")
# parser.add_argument('--server_address', type=str, default=server_address, help="Address of the Flower server")
# parser.add_argument('--partition_id', type=int, default=0, help="Partition ID for the client")
# parser.add_argument('--round', type=int, default=1, help="Current round number")
# args = parser.parse_args()

# # Load and preprocess data
# df = pd.read_csv('processed_data.csv')
# X = df.drop(columns=['diabetesMed_Yes']).values
# y = df['diabetesMed_Yes'].values
# y = (y > 0).astype(int)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split data into partitions
# partition_size = len(X) // num_clients
# partition_start = args.partition_id * partition_size
# partition_end = partition_start + partition_size if args.partition_id != num_clients - 1 else len(X)

# X_train, X_test, y_train, y_test = train_test_split(X[partition_start:partition_end], y[partition_start:partition_end], test_size=0.2, random_state=42)

# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
        
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# trainset = CustomDataset(X_train, y_train)
# testset = CustomDataset(X_test, y_test)

# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# net = create_model(input_size=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_size=output_dim, device=DEVICE)

# def train(model, dataloader, epochs):
#     print(f"Training in round {args.round} for client {args.partition_id}")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#     model.train()

#     for _ in range(epochs):
#         for X_batch, y_batch in dataloader:
#             X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()

# def test(model, dataloader):
#     criterion = nn.CrossEntropyLoss()
#     model.eval()
#     correct, total, test_loss = 0, 0, 0.0
#     all_labels, all_predictions = [], []

#     with torch.no_grad():
#         for X_batch, y_batch in dataloader:
#             X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             test_loss += loss.item() * y_batch.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += y_batch.size(0)
#             correct += (predicted == y_batch).sum().item()
#             all_labels.extend(y_batch.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())

#     accuracy = correct / total
#     precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
#     recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
#     f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)
    
#     return test_loss / total, accuracy, {"precision": precision, "recall": recall, "f1_score": f1}

# class DiabetesClient(NumPyClient):
#     def get_parameters(self):
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]

#     def fit(self, parameters, config):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)
#         train(net, trainloader, epochs=epochs)
#         return self.get_parameters(), len(trainset), {}

#     def evaluate(self, parameters, config):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)
#         loss, accuracy, metrics = test(net, testloader)
#         return float(loss), len(testset), {"accuracy": float(accuracy), **metrics}

# if __name__ == "__main__":
#     start_client(server_address=args.server_address, client=DiabetesClient().to_client())
