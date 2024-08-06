# import os
# import logging
# import numpy as np
# import torch
# from typing import List, Tuple, Dict, Any, Callable, Optional, Union
# from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
# from flwr.server.strategy import FedAvg
# from flwr.server.client_proxy import ClientProxy
# from model import create_model
# from flwr.common import FitRes
# from flwr.common import Metrics
# import yaml
# import traceback

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load configuration from YAML file
# def load_config(yaml_file: str) -> Dict[str, Any]:
#     try:
#         with open(yaml_file, 'r') as file:
#             config = yaml.safe_load(file)
#         return config
#     except Exception as e:
#         logging.error(f"Error loading config file {yaml_file}: {e}")
#         traceback.print_exc()

# config = load_config('configuracion.yaml')

# input_dim = config['model']['input_dim']
# hidden_dim = config['model']['hidden_dim']
# num_layers = config['model']['num_layers']
# output_dim = config['model']['output_dim']
# save_model_directory = config['model']['save_model_directory']

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def parameters_to_state_dict(aggregated_parameters, model):
#     try:
#         state_dict = model.state_dict()
#         parameter_tensors = parameters_to_ndarrays(aggregated_parameters)
#         for name, param in zip(state_dict.keys(), parameter_tensors):
#             state_dict[name] = torch.tensor(param)
#         return state_dict
#     except Exception as e:
#         logging.error("Error converting parameters to state_dict:", exc_info=True)
#         traceback.print_exc()

# def validate_parameters(aggregated_parameters):
#     try:
#         ndarrays = parameters_to_ndarrays(aggregated_parameters)
#         for ndarray in ndarrays:
#             logging.info("----------------------------------------------------------------")
#             logging.info(ndarray.shape)
#             logging.info(ndarray[:5])
#     except Exception as e:
#         logging.error("Error validating aggregated parameters:", exc_info=True)
#         traceback.print_exc()

# def validate_and_unpack_parameters(aggregated_parameters: Parameters):
#     try:
#         ndarrays = parameters_to_ndarrays(aggregated_parameters)
#         for i, ndarray in enumerate(ndarrays):
#             logging.info(f"Shape of ndarray {i}: {ndarray.shape}")
#         return ndarrays
#     except Exception as e:
#         logging.error("Error unpacking parameters:", exc_info=True)
#         traceback.print_exc()
#         return None

# def log_parameters(parameters: Parameters):
#     try:
#         for i, tensor in enumerate(parameters.tensors):
#             try:
#                 tensor_array = np.frombuffer(tensor, dtype=np.float32)
#                 logging.info(f"Tensor {i} shape: {tensor_array.shape}, values: {tensor_array[:5]}")  # Show first 5 values
#             except ValueError as ve:
#                 logging.error(f"ValueError while logging tensor {i}:", exc_info=True)
#                 traceback.print_exc()
#     except Exception as e:
#         logging.error("Error logging parameters:", exc_info=True)
#         traceback.print_exc()




# class FedNova(FedAvg):
#     def __init__(self, save_model_directory: str, learning_rate: float, epochs: int, handle_aggregated_parameters: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None):
#         super().__init__()
#         self.save_model_directory = save_model_directory
#         self.learning_rate = learning_rate
#         self.epochs = epochs
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
#             # Use learning_rate and epochs from the configuration
#             eta_i = self.learning_rate
#             t_i = self.epochs

#             # Log and validate client parameters
#             log_parameters(fit_res.parameters)
#             client_parameters = validate_and_unpack_parameters(fit_res.parameters)
#             if client_parameters is None:
#                 logging.error(f"Client {client} provided invalid parameters, skipping...")
#                 continue

#             # Normalize updates by the amount of work (eta_i * t_i)
#             normalized_updates = [param / (eta_i * t_i) for param in client_parameters]

#             if not aggregated_parameters_ndarrays:
#                 aggregated_parameters_ndarrays = [np.zeros_like(param) for param in normalized_updates]

#             for i, param in enumerate(normalized_updates):
#                 aggregated_parameters_ndarrays[i] += param * (num_examples_client / total_examples)

#         # Convert aggregated_parameters to state_dict
#         aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
#         state_dict = parameters_to_state_dict(aggregated_parameters, model)
        
#         # Create directory for the round
#         round_dir = os.path.join(self.save_model_directory, f'round_{server_round}')
#         os.makedirs(round_dir, exist_ok=True)

#         # Save the aggregated model
#         model.load_state_dict(state_dict)

#         # Export the model to TorchScript
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
#                 logging.info(f"  Loss: {res.loss}")
#                 logging.info(f"  Metrics: {res.metrics}")
#                 logging.info(f"  Number of examples: {res.num_examples}")

#             accuracies = [res.metrics.get("accuracy", 0) * res.num_examples for client, res in results]
#             num_examples = [res.num_examples for client, res in results]

#             aggregated_accuracy = sum(accuracies) / sum(num_examples) if num_examples else 0
#             logging.info(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

#             return aggregated_loss, {"accuracy": aggregated_accuracy}
#         except Exception as e:
#             logging.error("Error in aggregate_evaluate:", exc_info=True)
#             traceback.print_exc()
#             return None, {}  # Return empty metrics in case of failure




import os
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from model import create_model
from flwr.common import FitRes
from flwr.common import Metrics
import yaml
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from YAML file
def load_config(yaml_file: str) -> Dict[str, Any]:
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file {yaml_file}: {e}")
        traceback.print_exc()

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