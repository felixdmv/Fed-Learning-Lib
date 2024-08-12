import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os
import re
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from utils import load_config


# Función para ordenar naturalmente los nombres de archivos y carpetas
def natural_sort_key(s: str) -> List:
    """
    Generates a key for natural sorting of strings.

    Args:
        s (str): The string to generate the key for.

    Returns:
        List: The generated key for natural sorting.

    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


# Preprocesamiento de datos
def preprocess_data(csv_file: str):
    """
    Preprocesses the data from a CSV file.

    Parameters:
    - csv_file (str): The path to the CSV file.

    Returns:
    - X (numpy.ndarray): The preprocessed feature matrix.
    - y (numpy.ndarray): The preprocessed target vector.
    """
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['diabetesMed_Yes']).values
    y = df['diabetesMed_Yes'].values
    y = (y > 0).astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def split_data(X, y, test_size=0.4, random_state=2):
    """
    Splits the data into training and testing sets.

    Parameters:
    - X: The input features.
    - y: The target variable.
    - test_size: The proportion of the data to be used for testing. Default is 0.4.
    - random_state: The seed used by the random number generator. Default is 2.

    Returns:
    - X_train: The training set of input features.
    - X_test: The testing set of input features.
    - y_train: The training set of target variable.
    - y_test: The testing set of target variable.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

class CustomDataset(Dataset):
    """
    A custom dataset class for handling X and y data.
    Args:
        X (list or array-like): The input features.
        y (list or array-like): The target labels.
    Attributes:
        X (torch.Tensor): The input features as a torch tensor.
        y (torch.Tensor): The target labels as a torch tensor.
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
    """
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
    """
    Evaluates all models found in the specified directory using the given dataloader.
    Args:
        model_dir (str): The directory path where the models are located.
        dataloader (DataLoader): The dataloader used to evaluate the models.
    Returns:
        dict: A dictionary containing the evaluation results for each model found. The keys are the model names
        (extracted from the file names) and the values are lists of accuracy scores.
    """
    results = {}
    model_files_found = False

    for root, dirs, files in os.walk(model_dir):
        # Ordenar directorios y archivos naturalmente
        dirs.sort(key=natural_sort_key)
        files.sort(key=natural_sort_key)
        
        for file in files:
            if file.endswith('.pth'):
                model_files_found = True
                model_path = os.path.join(root, file)
                print(f"Evaluating model: {model_path}")
                try:
                    model = load_model(model_path)
                    metrics = evaluate_model(model, dataloader)
                    
                    match = re.match(r'(client_\d+|aggregated_model)', file)
                    if match:
                        model_name = match.group(1)
                        if model_name not in results:
                            results[model_name] = []
                        results[model_name].append(metrics["accuracy"])
                    
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


def plot_accuracy(results):
    """
    Plots the accuracy of each client over rounds.
    Parameters:
    - results (dict): A dictionary containing the accuracy results for each client.
    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    max_length = max(len(acc_list) for acc_list in results.values())

    for client, accuracies in results.items():
        plt.plot(range(len(accuracies)), accuracies, label=f'Accuracy {client}')
    
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    config = load_config('./config/configuracion.yaml')
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    output_dim = config['model']['output_dim']
    dropout_rate = config['training']['dropout_rate']
    batch_size = config['training']['batch_size']

    X, y = preprocess_data(config['client']['train_data_path'])
    X_train, X_test, y_train, y_test = split_data(X, y)
    testset = CustomDataset(X_test, y_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model_dir = './models'
    results = evaluate_all_models(model_dir, testloader)
    plot_accuracy(results)
