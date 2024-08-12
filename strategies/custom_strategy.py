import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from model import create_model
from utils import load_config
from flwr.common import FitRes
from flwr.common import Metrics
import yaml
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


config = load_config('./config/configuracion.yaml')

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

class Custom(FedAvg):
    def __init__(self, save_model_directory: str, handle_aggregated_parameters: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None):
        super().__init__()
        self.save_model_directory = save_model_directory
        self.global_parameters = None
        self.previous_model_parameters = None
        self.parameter_shapes = None
        self.handle_aggregated_parameters = handle_aggregated_parameters

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[int, Exception]]) -> Tuple[Parameters, Dict[str, float]]:
        num_clients = len(results)
        total_examples = sum(result.num_examples for _, result in results)

        # Initialize the model and get the state_dict
        model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)
        model_state_dict = model.state_dict()
        if self.parameter_shapes is None:
            self.parameter_shapes = {name: param.shape for name, param in model_state_dict.items()}

        aggregated_parameters_ndarrays = []

        for client, fit_res in results:
            num_examples_client = fit_res.num_examples

            # Log and validate client parameters
            log_parameters(fit_res.parameters)
            client_parameters = validate_and_unpack_parameters(fit_res.parameters)
            if client_parameters is None:
                logging.error(f"Client {client} provided invalid parameters, skipping...")
                continue

            if not aggregated_parameters_ndarrays:
                aggregated_parameters_ndarrays = [np.zeros_like(param) for param in client_parameters]

            for i, param in enumerate(client_parameters):
                aggregated_parameters_ndarrays[i] += param * (num_examples_client / total_examples)

        # Convert aggregated_parameters to state_dict
        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
        state_dict = parameters_to_state_dict(aggregated_parameters, model)
        # Crear directorio para la ronda
        round_dir = os.path.join(self.save_model_directory, f'round_{server_round}')
        os.makedirs(round_dir, exist_ok=True)

        # Guardar el modelo agregado
        model.load_state_dict(state_dict)

        # Exportar el modelo a TorchScript
        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))
        logging.info(f'Saved global model for round {server_round}: {model_scripted}.pth')

        # Return both aggregated parameters and a dictionary of metrics (if any)
        aggregated_metrics = {}  # Replace with actual metrics if needed
        return aggregated_parameters, aggregated_metrics

            
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation metrics from clients."""
        try:
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
        except Exception as e:
            logging.error("Error in aggregate_evaluate:", exc_info=True)
            traceback.print_exc()
            return None, {}  # Return empty metrics in case of failure





# class Custom(FedAvg):
#     def __init__(self, save_model_directory: str, handle_aggregated_parameters: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None):
#         super().__init__()
#         self.save_model_directory = save_model_directory
#         self.global_parameters = None
#         self.previous_model_parameters = None
#         self.parameter_shapes = None
#         self.handle_aggregated_parameters = handle_aggregated_parameters

#     def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Tuple[int, Exception]]) -> Tuple[Parameters, Dict[str, float]]:
#         num_clients = len(results)
#         total_examples = sum(result.num_examples for _, result in results)

#         # Initialize the model and get the state_dict
#         model = create_model(input_dim, hidden_dim, num_layers, output_dim, DEVICE)
#         model_state_dict = model.state_dict()
#         if self.parameter_shapes is None:
#             self.parameter_shapes = {name: param.shape for name, param in model_state_dict.items()}

#         aggregated_parameters_ndarrays = []

#         for client, fit_res in results:
#             num_examples_client = fit_res.num_examples

#             # Log and validate client parameters
#             log_parameters(fit_res.parameters)
#             client_parameters = validate_and_unpack_parameters(fit_res.parameters)
#             if client_parameters is None:
#                 logging.error(f"Client {client} provided invalid parameters, skipping...")
#                 continue

#             if not aggregated_parameters_ndarrays:
#                 aggregated_parameters_ndarrays = [np.zeros_like(param) for param in client_parameters]

#             for i, param in enumerate(client_parameters):
#                 aggregated_parameters_ndarrays[i] += param * (num_examples_client / total_examples)

#         # Convert aggregated_parameters to state_dict
#         aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
#         state_dict = parameters_to_state_dict(aggregated_parameters, model)
#         # Crear directorio para la ronda
#         round_dir = os.path.join(self.save_model_directory, f'round_{server_round}')
#         os.makedirs(round_dir, exist_ok=True)

#         # Guardar el modelo agregado
#         model.load_state_dict(state_dict)

#         # Exportar el modelo a TorchScript
#         model_scripted = torch.jit.script(model)
#         model_scripted.save(os.path.join(round_dir, 'aggregated_model.pth'))
#         logging.info(f'Saved global model for round {server_round}: {model_scripted}.pth')

#         # Return both aggregated parameters and a dictionary of metrics (if any)
#         aggregated_metrics = {}  # Replace with actual metrics if needed
#         return aggregated_parameters, aggregated_metrics

            
#     def aggregate_evaluate(
#         self,
#         server_round: int,
#         results: List[Tuple[ClientProxy, Metrics]],
#         failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
#     ) -> Tuple[Optional[float], Dict[str, float]]:
#         """Aggregate evaluation metrics from clients."""
#         try:
#             if not results:
#                 return None, {}

#             aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

#             for client, res in results:
#                 logging.info(f"Client {client} results:")
#                 logging.info(f"  Loss: {res['loss']}")
#                 logging.info(f"  Metrics: {res['metrics']}")
#                 logging.info(f"  Number of examples: {res['num_examples']}")

#             accuracies = [res['metrics'].get("accuracy", 0) * res['num_examples'] for client, res in results]
#             num_examples = [res['num_examples'] for client, res in results]

#             aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
#             logging.info(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

#             return aggregated_loss, {"accuracy": aggregated_accuracy}
#         except Exception as e:
#             logging.error("Error in aggregate_evaluate:", exc_info=True)
#             traceback.print_exc()
#             return None, {}  # Return empty metrics in case of failure


