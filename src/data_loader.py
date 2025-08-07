import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV dataset from the specified filepath.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame. Returns an empty DataFrame if file not found.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}. Please ensure the file exists.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return pd.DataFrame()
