# fedprox_strategy.py

import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
import logging
import os
import yaml
from typing import List, Tuple, Dict, Any, Union, Optional
from flwr.common import EvaluateRes, FitRes, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from model import create_model
from utils import load_config


# Load configuration
config = load_config('./config/configuracion.yaml')

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

class FedProx(fl.server.strategy.FedAvg):
    def __init__(self, mu: float, save_model_directory: str):
        super().__init__()
        self.mu = mu
        self.save_model_directory = save_model_directory
        self.global_parameters = None  # Assume it will be initialized elsewhere

    def aggregate_fit(self, rnd: int, results: List[Tuple[Any, Dict[str, torch.Tensor]]], failures: List[Tuple[int, Exception]]) -> Dict[str, torch.Tensor]:
        try:
            # Aggregate local parameters
            aggregated_parameters = super().aggregate_fit(rnd, results, failures)
            
            # Create directory for the round
            round_dir = os.path.join(self.save_model_directory, f'round_{rnd}')
            os.makedirs(round_dir, exist_ok=True)
            
            # Create a model and check parameter sizes
            model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)
            state_dict = parameters_to_state_dict(aggregated_parameters, model)
            model.load_state_dict(state_dict)
            
            # Export the model to TorchScript
            model_scripted = torch.jit.script(model)
            model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))
            logging.info(f'Saved global model for round {rnd}: {model_scripted}.pth')
        
            self.previous_model_parameters = aggregated_parameters

            return aggregated_parameters

        except Exception as e:
            logging.error(f'Error during aggregation or model saving: {e}')
            raise

    def configure_fit(self, server_round: int, parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with Prox penalty."""
        config = {
            "mu": self.mu,  # Add the mu parameter for Prox penalty
        }
        return [(client, FitIns(parameters, config)) for client in client_manager.clients.values()]

    def _weighted_average(self, parameters_list: List[Tuple[List[np.ndarray], int]], num_examples: np.ndarray) -> List[np.ndarray]:
        """Compute weighted average of parameters."""
        if not parameters_list:
            raise ValueError("No parameters to aggregate.")

        num_parameters = len(parameters_list[0][0])
        aggregated_weights = [np.zeros_like(params[0]) for params in parameters_list[0][0]]
        total_examples = np.sum(num_examples)

        for params, num_examples_client in zip(parameters_list, num_examples):
            for i in range(num_parameters):
                aggregated_weights[i] += params[0][i] * num_examples_client

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

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        for client, res in results:
            print(f"Client {client} results:")
            print(f"  Loss: {res.loss}")
            print(f"  Metrics: {res.metrics}")
            print(f"  Number of examples: {res.num_examples}")

        accuracies = [res.metrics.get("accuracy", 0) * res.num_examples for _, res in results]
        num_examples = [res.num_examples for _, res in results]

        aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        return aggregated_loss, {"accuracy": aggregated_accuracy}
