import json
import os

def save_results(results: dict, filepath: str):
    """
    Saves a dictionary of results to a JSON file.

    Args:
        results (dict): The dictionary containing results.
        filepath (str): The path to the JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")

def load_results(filepath: str) -> dict:
    """
    Loads results from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The loaded dictionary of results. Returns empty dict if file not found.
    """
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        print(f"Results loaded from {filepath}")
        return results
    except FileNotFoundError:
        print(f"Error: Results file not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. File might be corrupted.")
        return {}
