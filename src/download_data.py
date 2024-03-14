import os
import kaggle
import sys

def download_dataset(dataset: str, path: str = './data'):
    """
    Download a dataset from Kaggle.

    :param dataset: The dataset path on Kaggle (e.g., 'vencerlanz09/100-bird-species').
    :param path: Local path to save the dataset.
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    # Use the Kaggle API to download the dataset
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=path, unzip=True)

    print(f"Dataset downloaded to: {path}")

if __name__ == "__main__":
    # Define the dataset path on Kaggle
    dataset_path = sys.argv

    # Define the local directory to save the dataset
    download_path = '../data'

    # Download the dataset
    download_dataset(dataset_path, download_path)
