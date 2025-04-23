from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokenised_dataset

dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = 500
min_frequency = 0  # Set the minimum frequency for tokenization

def main():
    dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dataset)
    get_tokenised_dataset(clean_dataset, min_frequency)

if __name__ == "__main__":
    main()