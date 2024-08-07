# utils.py

from typing import List, Tuple, Dict
import yaml
import logging
import traceback
from typing import Any

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


# Load configuration from YAML file
def load_config(yaml_file: str) -> Dict[str, Any]:
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file {yaml_file}: {e}")
        traceback.print_exc()