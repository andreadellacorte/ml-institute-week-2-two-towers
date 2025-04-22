import json
from datasets import load_dataset

def extract_eda_lines(dataset, output_file: str, num_lines=None):
    """
    Extracts  lines from the Hugging Face dataset `microsoft/ms_marco` and saves them to the output file.

    Args:
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, entry in enumerate(dataset):
            if num_lines is not None and i >= num_lines:
                break
            json.dump(entry, outfile)
            outfile.write("\n")

    print(f"Extracted {'all' if num_lines is None else num_lines} lines from the Hugging Face dataset to {output_file}.")

def main():
    dataset_id = "microsoft/ms_marco"  # Replace with your input file path
    dataset_version = "v1.1"  # Replace with your desired dataset version

    dataset = load_dataset(dataset_id, dataset_version, split='train')

    output_file = 'data/raw/ms_marco_eda.txt'  # Replace with your desired output file path

    extract_eda_lines(dataset, output_file)

if __name__ == "__main__":
    main()