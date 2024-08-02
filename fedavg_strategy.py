# fedavg_strategy.py

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, start_server
import torch
import numpy as np
from collections import OrderedDict
import logging
import os
import yaml
from typing import List, Tuple, Dict, Any
from torch import nn
from flwr.common import EvaluateRes, FitRes
from typing import Union, Optional
from flwr.server.client_proxy import ClientProxy
from model import create_model


# Load configuration from YAML file
def load_config(yaml_file: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config('configuracion.yaml')

input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
output_dim = config['model']['output_dim']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parameters_to_state_dict(aggregated_parameters, model):
    """Convert aggregated parameters to state_dict format."""
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in aggregated_parameters:
            state_dict[name] = torch.tensor(aggregated_parameters[name])
    return state_dict

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, save_model_directory: str):
         super().__init__()
         self.save_model_directory = save_model_directory
         self.global_parameters = None  # Asume que se inicializará en algún otro momento


    def aggregate_fit(self, rnd: int, results: List[Tuple[Any, Dict[str, torch.Tensor]]], failures: List[Tuple[int, Exception]]) -> Dict[str, torch.Tensor]:
        # Debugging: Imprime los resultados recibidos de los clientes
        for client_id, result in results:
            print(f"Client {client_id} sent parameters:")
            #print(f"Parameters: {result.parameters}")  # Acceso adecuado a los atributos
            if result.parameters is not None:
                for param in result.parameters.tensors:
                    param_array = np.frombuffer(param, dtype=np.float32)
                    param_shape = param_array.shape
                    param_size = param_array.size
                    print(f"Parameter shape: {param_shape}")
                    print(f"Parameter size: {param_size}")
            print(f"Num examples: {result.num_examples}")
            print(f"Metrics: {result.metrics}")
        try:            
            # Agregar parámetros locales
            aggregated_parameters = super().aggregate_fit(rnd, results, failures)
            
            # Crear directorio para la ronda
            round_dir = os.path.join(self.save_model_directory, f'round_{rnd}')
            os.makedirs(round_dir, exist_ok=True)
            
            
            # Crear un modelo y verificar los tamaños de los parámetros
            model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)

            for name, param in model.state_dict().items():
                print(f"{name}: {param.size()}")  # Asegúrate de que esto coincide con lo esperado

            
            # Convertir parámetros agregados a state_dict
            state_dict = parameters_to_state_dict(aggregated_parameters, model)
            
            # Cargar el state_dict en el modelo
            model.load_state_dict(state_dict)
            
            # Exportar el modelo a TorchScript
            model_scripted = torch.jit.script(model)
            model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))
            logging.info(f'Saved global model for round {rnd}: {model_scripted}.pth')
        
            self.previous_model_parameters = aggregated_parameters

            return aggregated_parameters

        except Exception as e:
            logging.error(f'Error during aggregation or model saving: {e}')
            raise

    def _weighted_average(self, parameters_list: List[Tuple[List[np.ndarray], int]], num_examples: np.ndarray) -> List[np.ndarray]:
        """Compute weighted average of parameters."""
        if not parameters_list:
            raise ValueError("No parameters to aggregate.")

        # Get the number of parameter sets and initialize aggregated weights
        num_parameters = len(parameters_list[0][0])
        aggregated_weights = [np.zeros_like(params[0]) for params in parameters_list[0][0]]

        total_examples = np.sum(num_examples)

        for params, num_examples_client in zip(parameters_list, num_examples):
            for i in range(num_parameters):
                aggregated_weights[i] += params[0][i] * num_examples_client

        # Normalize aggregated weights
        for i in range(num_parameters):
            aggregated_weights[i] /= total_examples

        return aggregated_weights



    def _save_model(self, rnd: int, aggregated_weights: List[np.ndarray]):
        """Save the aggregated model."""
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.parameters.keys(), aggregated_weights)})
        model_path = os.path.join(self.save_model_directory, f"model_round_{rnd}.pth")
        torch.save(state_dict, model_path)
        logging.info(f"Saved model for round {rnd} to {model_path}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation metrics from clients."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Debug: Print structure of EvaluateRes
        for client, res in results:
            print(f"Client {client} results:")
            print(f"  Loss: {res.loss}")
            print(f"  Metrics: {res.metrics}")
            print(f"  Number of examples: {res.num_examples}")

        # Extract accuracy metrics and number of examples from results
        accuracies = [res.metrics.get("accuracy", 0) * res.num_examples for _, res in results]
        num_examples = [res.num_examples for _, res in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


    
    def get_history(self):
        return self.history
