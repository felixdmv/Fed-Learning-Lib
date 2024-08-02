# abstract_strategy.py

from abc import ABC, abstractmethod
import flwr as fl
from flwr.common import Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import List, Tuple, Dict, Union, Optional

class CustomStrategy(fl.server.strategy.Strategy, ABC):
    """Custom strategy for federated learning."""
    
    @abstractmethod
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize the (global) model parameters."""
        pass

    @abstractmethod
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        pass

    @abstractmethod
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        pass

    @abstractmethod
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        pass

    @abstractmethod
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        pass

    @abstractmethod
    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters."""
        pass
