import flwr as fl
import os
from typing import Dict, Any
from flwr.common import Parameters, parameters_to_ndarrays
from strategies.fedavg_strategy import FedAvg
from strategies.fedprox_strategy import FedProx
from strategies.fednova_strategy import FedNova
from strategies.custom_strategy import Custom
from strategies.scaffold_strategy import Scaffold
from utils import load_config
import traceback


configuracion = load_config('./config/configuracion.yaml')

strategy_name = configuracion['training']['strategy']
learning_rate = configuracion['training']['learning_rate']
epochs = configuracion['training']['epochs']
local_steps = configuracion['training']['local_steps']
prox_mu = configuracion['training']['prox_mu']

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
                prox_mu=prox_mu,
                save_model_directory=save_model_directory
            )
        
        elif strategy_name == 'fedNova':
            return FedNova(
                save_model_directory=save_model_directory, 
                handle_aggregated_parameters=handle_aggregated_parameters, 
                learning_rate=learning_rate, 
                epochs=epochs)
    
        elif strategy_name == 'custom':
            return Custom(
                save_model_directory=save_model_directory,
                handle_aggregated_parameters=handle_aggregated_parameters
            )
        
        elif strategy_name == 'scaffold':
            return Scaffold(
                save_model_directory=save_model_directory,
                learning_rate=learning_rate,
                local_steps=local_steps
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
    """
    Starts the server for federated learning.
    Args:
        config (dict): A dictionary containing the configuration parameters for the server.
    Raises:
        Exception: If there is an error starting the server.
    """
    pass
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
        config = load_config('./config/configuracion.yaml')
        start_server(config)
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
