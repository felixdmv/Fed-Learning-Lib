import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from model import create_model
from utils import load_config, parameters_to_ndarrays
from flwr.common import FitRes, EvaluateRes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = load_config('./config/configuracion.yaml')

input_dim = config['model']['input_dim']
hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
output_dim = config['model']['output_dim']
save_model_directory = config['model']['save_model_directory']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Scaffold(FedAvg):
    def __init__(self, save_model_directory: str, learning_rate: float, local_steps: int):
        super().__init__()
        self.save_model_directory = save_model_directory
        self.learning_rate = learning_rate
        self.local_steps = local_steps
        self.global_control = None

    def initialize_global_control(self, model_parameters):
        self.global_control = [np.zeros_like(param) for param in model_parameters]

    def aggregate_fit(self, server_round: int, results: List[Tuple[Any, FitRes]], failures: List[Tuple[int, Exception]]) -> Tuple[Parameters, Dict[str, float]]:
        num_clients = len(results)
        if num_clients == 0:
            raise ValueError("No clients available for aggregation")

        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)
        model_state_dict = model.state_dict()

        aggregated_parameters_ndarrays = [np.zeros_like(param.cpu().numpy()) for param in model_state_dict.values()]
        aggregated_control_ndarrays = [np.zeros_like(param.cpu().numpy()) for param in model_state_dict.values()]

        for client, fit_res in results:
            num_examples_client = fit_res.num_examples

            # Convert Parameters object to list of ndarrays
            client_parameters = parameters_to_ndarrays(fit_res.parameters)

            # Handle the case where control_params might be missing
            if 'control_params' in fit_res.metrics:
                client_control_params = parameters_to_ndarrays(fit_res.metrics['control_params'])
            else:
                logging.warning(f"Client {client} did not send control_params. Using zeros as default.")
                client_control_params = [np.zeros_like(param) for param in client_parameters]

            for i, param in enumerate(client_parameters):
                aggregated_parameters_ndarrays[i] += param * (num_examples_client / total_examples)

            for i, control_param in enumerate(client_control_params):
                aggregated_control_ndarrays[i] += control_param * (num_examples_client / total_examples)

        aggregated_parameters = [torch.tensor(param).to(DEVICE) for param in aggregated_parameters_ndarrays]
        aggregated_control_params = [torch.tensor(param).to(DEVICE) for param in aggregated_control_ndarrays]

        aggregated_parameters_dict = {name: param for name, param in zip(model_state_dict.keys(), aggregated_parameters)}
        aggregated_control_params_dict = {name: param for name, param in zip(model_state_dict.keys(), aggregated_control_params)}

        model.load_state_dict(aggregated_parameters_dict)

        round_dir = os.path.join(self.save_model_directory, f'round_{server_round}')
        os.makedirs(round_dir, exist_ok=True)

        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))
        logging.info(f'Saved global model for round {server_round}: {model_scripted}.pth')

        self.previous_model_parameters = aggregated_parameters_dict
        self.previous_control_params = aggregated_control_params_dict

        # Convert aggregated parameters back to a Parameters object
        aggregated_parameters_obj = ndarrays_to_parameters([param.cpu().numpy() for param in aggregated_parameters])

        return aggregated_parameters_obj, {}



    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Tuple[int, Exception]]) -> Tuple[Optional[float], Dict[str, float]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracies = [res.metrics.get("accuracy", 0) * res.num_examples for _, res in results]
        num_examples = [res.num_examples for _, res in results]
        aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
        return aggregated_loss, {"accuracy": aggregated_accuracy}