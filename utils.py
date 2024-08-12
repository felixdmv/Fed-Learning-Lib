# utils.py

from typing import List, Tuple, Dict
from flwr.common import parameters_to_ndarrays
import torch
import yaml
import logging
import traceback
from typing import Any
from flwr.common import Parameters
import numpy as np

# Load configuration from YAML file
def load_config(yaml_file: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        yaml_file (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The contents of the YAML configuration file as a dictionary.

    Raises:
        Exception: If there is an error loading the configuration file.
    """
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file {yaml_file}: {e}")
        traceback.print_exc()


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate metrics by computing the weighted average.

    Args:
        metrics (List[Tuple[int, Dict[str, float]]]): A list where each item is a tuple containing:
            - The number of examples processed by the client.
            - A dictionary of metrics, e.g., {"accuracy": 0.85}.
    
    Returns:
        Dict[str, float]: A dictionary containing the aggregated metrics.
    """
    try:
        # Check if metrics list is empty
        if not metrics:
            return {"accuracy": 0}

        # Initialize sums and counts
        total_accuracy = 0
        total_examples = 0

        # Accumulate weighted metrics
        for num_examples, metrics_dict in metrics:
            accuracy = metrics_dict.get("accuracy", 0)
            total_accuracy += accuracy * num_examples
            total_examples += num_examples

        # Compute weighted average accuracy
        aggregated_accuracy = total_accuracy / total_examples if total_examples > 0 else 0

        return {"accuracy": aggregated_accuracy}

    except Exception as e:
        # Log any exceptions and return a default value
        import logging
        logging.error(f"Error during metrics aggregation: {e}")
        return {"accuracy": 0}




def parameters_to_state_dict(aggregated_parameters, model):
    """
    Converts aggregated parameters to a state dictionary for a given model.

    Args:
        aggregated_parameters (list): A list of aggregated parameters.
        model (torch.nn.Module): The model for which the state dictionary is generated.

    Returns:
        dict: The state dictionary containing the converted parameters.

    Raises:
        Exception: If there is an error converting the parameters to state dictionary.
    """
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
    """
    Validates the aggregated parameters.

    Args:
        aggregated_parameters: The aggregated parameters to be validated.

    Raises:
        Exception: If an error occurs while validating the parameters.

    Returns:
        None
    """
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
    """
    Validates and unpacks the aggregated parameters.

    Args:
        aggregated_parameters (Parameters): The aggregated parameters to be validated and unpacked.

    Returns:
        List[np.ndarray] or None: A list of numpy arrays representing the unpacked parameters, or None if an error occurred.

    Raises:
        None

    """
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
    """
    Logs the parameters of a given Parameters object.

    Parameters:
        parameters (Parameters): The Parameters object containing the tensors to be logged.

    Raises:
        ValueError: If there is an error while logging a tensor.
        Exception: If there is an error while logging the parameters.

    """
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