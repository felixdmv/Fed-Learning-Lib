import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from model import create_model
from utils import load_config
from flwr.common import FitRes, FitIns, EvaluateRes
import yaml
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



config = load_config('configuracion.yaml')

input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
output_dim = config['model']['output_dim']
save_model_directory = config['model']['save_model_directory']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parameters_to_state_dict(aggregated_parameters, model):
    try:
        state_dict = model.state_dict()
        parameter_tensors = parameters_to_ndarrays(aggregated_parameters)
        for name, param in zip(state_dict.keys(), parameter_tensors):
            state_dict[name] = torch.tensor(param)
        return state_dict
    except Exception as e:
        logging.error("Error converting parameters to state_dict:", exc_info=True)
        traceback.print_exc()

def validate_parameters(aggregated_parameters):
    try:
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        for ndarray in ndarrays:
            logging.info("----------------------------------------------------------------")
            logging.info(ndarray.shape)
            logging.info(ndarray[:5])
    except Exception as e:
        logging.error("Error validating aggregated parameters:", exc_info=True)
        traceback.print_exc()

def validate_and_unpack_parameters(aggregated_parameters: Parameters):
    try:
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        for i, ndarray in enumerate(ndarrays):
            logging.info(f"Shape of ndarray {i}: {ndarray.shape}")
        return ndarrays
    except Exception as e:
        logging.error("Error unpacking parameters:", exc_info=True)
        traceback.print_exc()
        return None

def log_parameters(parameters: Parameters):
    try:
        for i, tensor in enumerate(parameters.tensors):
            try:
                tensor_array = np.frombuffer(tensor, dtype=np.float32)
                logging.info(f"Tensor {i} shape: {tensor_array.shape}, values: {tensor_array[:5]}")  # Show first 5 values
            except ValueError as ve:
                logging.error(f"ValueError while logging tensor {i}:", exc_info=True)
                traceback.print_exc()
    except Exception as e:
        logging.error("Error logging parameters:", exc_info=True)
        traceback.print_exc()

class Scaffold(FedAvg):
    def __init__(self, save_model_directory: str, learning_rate: float, local_steps: int):
        super().__init__()
        self.save_model_directory = save_model_directory
        self.learning_rate = learning_rate
        self.local_steps = local_steps
        self.global_control = None

    def initialize_global_control(self, model_parameters):
        self.global_control = [np.zeros_like(param) for param in model_parameters]

    def configure_fit(self, server_round: int, parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        config = {"local_steps": self.local_steps, "learning_rate": self.learning_rate, "global_control": self.global_control}
        return [(client, FitIns(parameters, config)) for client in client_manager.clients.values()]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[int, Exception]]) -> Parameters:
        if not results:
            return super().aggregate_fit(server_round, results, failures)

        if self.global_control is None:
            self.initialize_global_control(parameters_to_ndarrays(results[0][1].parameters))

        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        new_global_control = [np.zeros_like(control) for control in self.global_control]
        for client, fit_res in results:
            client_control = parameters_to_ndarrays(fit_res.metrics["client_control"])
            for i, (control, new_control) in enumerate(zip(client_control, new_global_control)):
                new_global_control[i] += (control - self.global_control[i]) / len(results)

        self.global_control = new_global_control

        self._save_model(server_round, aggregated_parameters)
        return aggregated_parameters

    def _save_model(self, rnd: int, aggregated_parameters):
        model = create_model(config['model']['input_dim'], config['model']['hidden_dim'], config['model']['num_layers'], config['model']['output_dim'], torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        state_dict = {name: torch.tensor(param) for name, param in zip(model.state_dict().keys(), parameters_to_ndarrays(aggregated_parameters))}
        model.load_state_dict(state_dict)
        model_scripted = torch.jit.script(model)
        round_dir = os.path.join(self.save_model_directory, f'round_{rnd}')
        os.makedirs(round_dir, exist_ok=True)
        model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Tuple[int, Exception]]) -> Tuple[Optional[float], Dict[str, float]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracies = [res.metrics.get("accuracy", 0) * res.num_examples for _, res in results]
        num_examples = [res.num_examples for _, res in results]
        aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
        return aggregated_loss, {"accuracy": aggregated_accuracy}