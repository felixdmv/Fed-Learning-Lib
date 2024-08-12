import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from flwr.common import Parameters, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from model import create_model
from utils import load_config, parameters_to_state_dict, log_parameters, validate_and_unpack_parameters
from flwr.common import FitRes
from flwr.common import Metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = load_config('./config/configuracion.yaml')

input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
output_dim = config['model']['output_dim']
save_model_directory = config['model']['save_model_directory']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedNova(FedAvg):
    def __init__(self, save_model_directory: str, learning_rate: float, epochs: int, handle_aggregated_parameters: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None):
        super().__init__()
        self.save_model_directory = save_model_directory
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.global_parameters = None
        self.previous_model_parameters = None
        self.parameter_shapes = None
        self.handle_aggregated_parameters = handle_aggregated_parameters

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[int, Exception]]) -> Tuple[Parameters, Dict[str, float]]:
        num_clients = len(results)
        total_examples = sum(result.num_examples for _, result in results)

        model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)
        model_state_dict = model.state_dict()
        if self.parameter_shapes is None:
            self.parameter_shapes = {name: param.shape for name, param in model_state_dict.items()}

        aggregated_parameters_ndarrays = []

        for client, fit_res in results:
            num_examples_client = fit_res.num_examples
            eta_i = self.learning_rate
            t_i = self.epochs

            log_parameters(fit_res.parameters)
            client_parameters = validate_and_unpack_parameters(fit_res.parameters)
            if client_parameters is None:
                logging.error(f"Client {client} provided invalid parameters, skipping...")
                continue

            normalized_updates = [param / (eta_i * t_i) for param in client_parameters]
            print(f"Normalized updates for client {client}: {normalized_updates}")

            if not aggregated_parameters_ndarrays:
                aggregated_parameters_ndarrays = [np.zeros_like(param) for param in normalized_updates]

            for i, param in enumerate(normalized_updates):
                aggregated_parameters_ndarrays[i] += param * (num_examples_client / total_examples)

        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
        state_dict = parameters_to_state_dict(aggregated_parameters, model)
        print(f"Aggregated state_dict: {state_dict}")
        
        round_dir = os.path.join(self.save_model_directory, f'round_{server_round}')
        os.makedirs(round_dir, exist_ok=True)

        model.load_state_dict(state_dict)
        self._validate_model_outputs(model)

        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))
        logging.info(f'Saved global model for round {server_round}: {model_scripted}.pth')

        aggregated_metrics = {}
        return aggregated_parameters, aggregated_metrics

    def _validate_model_outputs(self, model):
        dummy_input = torch.randn(1, input_dim).to(DEVICE)
        output = model(dummy_input)
        if not torch.all((output >= 0) & (output <= 1)):
            logging.error(f"Model outputs are not in [0, 1]: {output}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        for client, res in results:
            logging.info(f"Client {client} results:")
            logging.info(f"  Loss: {res.loss}")
            logging.info(f"  Metrics: {res.metrics}")
            logging.info(f"  Number of examples: {res.num_examples}")

        accuracies = [res.metrics.get("accuracy", 0) * res.num_examples for client, res in results]
        num_examples = [res.num_examples for client, res in results]

        aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
        logging.info(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        return aggregated_loss, {"accuracy": aggregated_accuracy}