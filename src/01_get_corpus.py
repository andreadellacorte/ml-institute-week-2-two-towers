from utils.data_utils import get_raw_dataset, get_clean_dataset, get_corpus

dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = 500
min_frequency = None  # Set the minimum frequency for tokenization

def main():
    dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dataset)
    get_corpus(clean_dataset)

if __name__ == "__main__":
    main()