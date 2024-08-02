import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os
from typing import Dict, Any

# Cargar la configuración desde el archivo YAML
def load_config(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Definir el modelo
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, dropout_rate):
        super(SimpleBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Convertir la salida a probabilidad
        return x

# Preprocesamiento de datos
def preprocess_data(csv_file: str):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['diabetesMed_Yes']).values
    y = df['diabetesMed_Yes'].values
    y = (y > 0).astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def split_data(X, y, test_size=0.4, random_state=2):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_model(model_path: str) -> nn.Module:
    """Cargar un modelo TorchScript desde el archivo especificado."""
    print(f"Loading model from {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()  # Configura el modelo en modo de evaluación
    return model

def evaluate_model(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    """Evalúa el modelo y devuelve métricas."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            # Usar torch.sigmoid() para obtener probabilidades y luego convertir a clases
            preds = (outputs > 0.5).long().squeeze()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def evaluate_all_models(model_dir: str, dataloader: DataLoader):
    results = {}
    model_files_found = False
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pth'):  
                model_files_found = True
                model_path = os.path.join(root, file)
                print(f"Evaluating model: {model_path}")
                try:
                    model = load_model(model_path)
                    metrics = evaluate_model(model, dataloader)
                    results[model_path] = metrics
                    print(f"Model: {model_path}")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1 Score: {metrics['f1_score']:.4f}")
                    print("-" * 50)
                except Exception as e:
                    print(f"Error loading or evaluating model {model_path}: {e}")
    if not model_files_found:
        print("No model files found in the directory.")
    return results

if __name__ == "__main__":
    config = load_config('configuracion.yaml')
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    output_dim = config['model']['output_dim']
    dropout_rate = config['training']['dropout_rate']
    batch_size = config['training']['batch_size']

    X, y = preprocess_data('processed_data.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    testset = CustomDataset(X_test, y_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model_dir = './models'
    results = evaluate_all_models(model_dir, testloader)
