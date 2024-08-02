import flwr as fl
import os
import yaml
from typing import Dict, Any
from io import BytesIO
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from fedavg_strategy import FedAvg
from fedprox_strategy import FedProx
from fednova_strategy import FedNova
import traceback

def load_config(yaml_file: str) -> Dict[str, Any]:
    print(f"Attempting to load configuration from {yaml_file}...")
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        traceback.print_exc()
        raise

configuracion = load_config('configuracion.yaml')

strategy_name = configuracion['training']['strategy']

def create_strategy(strategy_name: str, model_config: Dict[str, Any], training_config: Dict[str, Any], server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
    """Create an instance of the federated learning strategy based on the strategy name."""
    save_model_directory = model_config["save_model_directory"]
    os.makedirs(os.path.dirname(save_model_directory), exist_ok=True)
    
    print(f"Creating strategy: {strategy_name}...")
    print(f"Model configuration: {model_config}")
    print(f"Training configuration: {training_config}")
    print(f"Server configuration: {server_config}")
    
    try:
        if strategy_name == 'fedAvg':
            return FedAvg(
                save_model_directory=save_model_directory
            )
        
        elif strategy_name == 'fedProx':
            return FedProx(
                mu=training_config.get("prox_mu", 0.9),
                save_model_directory=save_model_directory
            )
        
        elif strategy_name == 'fedNova':
            return FedNova(
                save_model_directory=save_model_directory,
                handle_aggregated_parameters=handle_aggregated_parameters
            )
        else:
            raise ValueError(f"Strategy {strategy_name} not supported")
    except Exception as e:
        print(f"Error creating strategy {strategy_name}: {e}")
        traceback.print_exc()
        raise

def handle_aggregated_parameters(aggregated_parameters: Parameters) -> None:
    """Handle aggregated parameters from clients."""
    try:
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        for ndarray in ndarrays:
            print(f"Shape of ndarray: {ndarray.shape}")
        print(f"Aggregated parameters: {aggregated_parameters}")
    except Exception as e:
        print(f"Error handling aggregated parameters: {e}")
        traceback.print_exc()

def start_server(config):
    print("Starting server...")
    try:
        model_config = config['model']
        training_config = config['training']
        server_config = config['server']
        
        strategy = create_strategy(strategy_name, model_config, training_config, server_config)
        print(f"Strategy {strategy_name} created. Starting server...")
        
        fl.server.start_server(
            server_address=server_config["server_address"],
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=server_config["num_rounds"])
        )
        
    except Exception as e:
        print(f"Error starting the server: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Loading configuration...")
    try:
        config = load_config('configuracion.yaml')
        start_server(config)
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
