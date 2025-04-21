import os
from datasets import load_dataset
import json

def get_raw_dataset(dataset_id, dataset_version, max_lines=None):
    cache_file = f"data/raw/ms_marco_all.txt"  # Path to the cached dataset file

    if not os.path.exists(cache_file):
        print(f"Cache file not found. Downloading and saving dataset to {cache_file}...")
        dataset = load_dataset(dataset_id, dataset_version, split='train')
        with open(cache_file, "w") as f:
            for item in dataset:
                f.write(f"{item}\n")
    else:
        print(f"Cache file found at {cache_file}. Using cached dataset.")

    # Proceed with loading the dataset from the cache file or using it directly
    with open(cache_file, "r") as f:
        dataset = f.readlines()

    # Cap the number of lines if max_lines is specified
    if max_lines is not None:
        dataset = dataset[:max_lines]

    return dataset

def get_clean_dataset(dataset):
    # Embed the number of lines in the file name
    num_lines = len(dataset)
    cache_file = f"data/processed/ms_marco_clean_{num_lines}_lines.json"

    if not os.path.exists(cache_file):
        print(f"Cache file not found. Cleaning dataset and saving to {cache_file}...")
        cleaned_data = []
        for item in dataset:
            # Assuming the dataset is a JSON-like string with ' as a delimiter, parse it
            # only keep the query and passages fields
            
        
        with open(cache_file, "w") as f:
            json.dump(cleaned_data, f, indent=4)
    else:
        print(f"Cache file found at {cache_file}. Using cached cleaned dataset.")

    with open(cache_file, "r") as f:
        cleaned_dataset = json.load(f)

    return cleaned_dataset
   

def main():
    dataset_id = "microsoft/ms_marco"  # Replace with your input file path
    dataset_version = "v1.1"  # Replace with your desired dataset version

    max_lines = 1000

    dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dataset)

if __name__ == "__main__":
    main()