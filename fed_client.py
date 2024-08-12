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
from utils import load_config
import traceback
import logging



# Load configuration
config = load_config('./config/configuracion.yaml')

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
#df = pd.read_csv('processed_data.csv')
df = pd.read_csv(config['client']['train_data_path'])
X = df.drop(columns=['diabetesMed_Yes']).values
y = df['diabetesMed_Yes'].values
y = (y > 0).astype(int)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into partitions
partition_size = len(X) // num_clients
partition_start = args.partition_id * partition_size
partition_end = partition_start + partition_size if args.partition_id != num_clients - 1 else len(X)

X_train, X_test, y_train, y_test = train_test_split(X[partition_start:partition_end], y[partition_start:partition_end], test_size=0.2, random_state=2)

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
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logging.error("Error setting parameters:", exc_info=True)
            traceback.print_exc()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(epochs):
            for batch in self.trainloader:
                X_batch, y_batch = batch
                y_batch = y_batch.unsqueeze(1)  # Ensure target size is [batch_size, 1]
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.float())
                loss.backward()
                self.optimizer.step()

        datetime = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        model_scripted = torch.jit.script(self.model)
        model_file_path = os.path.join(save_model_directory, f"client_{args.partition_id}_{datetime}.pth")
        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_file_path)
        model_scripted.save(model_file_path)
        print(f"Model saved to {model_file_path}")

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





# class Client(NumPyClient):    
#     def __init__(self, model, trainloader, testloader):
#         self.model = model
#         self.trainloader = trainloader
#         self.testloader = testloader
#         self.criterion = nn.BCELoss()
#         self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

#     def get_parameters(self, config) -> List[np.ndarray]:
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]

#     def set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.train()

#         for epoch in range(epochs):
#             for batch in self.trainloader:
#                 X_batch, y_batch = batch
#                 y_batch = y_batch.unsqueeze(1)  # Ensure target size is [batch_size, 1]
#                 X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

#                 self.optimizer.zero_grad()
#                 outputs = self.model(X_batch)
#                 loss = self.criterion(outputs, y_batch.float())
#                 loss.backward()
#                 self.optimizer.step()

#         datetime = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
#         model_scripted = torch.jit.script(self.model)
#         model_file_path = os.path.join(save_model_directory, f"client_{args.partition_id}_{datetime}.pth")
#         os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
#         torch.save(self.model.state_dict(), model_file_path)
#         model_scripted.save(model_file_path)
#         print(f"Model saved to {model_file_path}")

#         return self.get_parameters(config), len(self.trainloader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()

#         loss = 0.0
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels = []

#         with torch.no_grad():
#             for X_batch, y_batch in self.testloader:
#                 X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
#                 y_batch = y_batch.unsqueeze(1)
#                 outputs = self.model(X_batch)
#                 loss += self.criterion(outputs, y_batch.float()).item()
#                 predicted = (outputs > 0.5).int()
#                 total += y_batch.size(0)
#                 correct += (predicted == y_batch).sum().item()
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(y_batch.cpu().numpy())

#         accuracy = correct / total
#         precision = precision_score(all_labels, all_preds)
#         recall = recall_score(all_labels, all_preds)
#         f1 = f1_score(all_labels, all_preds)
#         loss /= len(self.testloader.dataset)

#         metrics = {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "loss": loss
#         }

#         return float(loss), len(self.testloader.dataset), metrics


