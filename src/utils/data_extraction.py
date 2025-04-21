import json
from datasets import load_dataset

def extract_5k_lines(dataset, output_file: str):
    """
    Extracts the first 5000 lines from the Hugging Face dataset `microsoft/ms_marco` and saves them to the output file.

    Args:
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, entry in enumerate(dataset):
            if i >= 5000:
                break
            json.dump(entry, outfile)
            outfile.write("\n")

    print(f"Extracted 5000 lines from the Hugging Face dataset to {output_file}.")

def main():
    dataset_id = "microsoft/ms_marco"  # Replace with your input file path
    dataset_version = "v1.1"  # Replace with your desired dataset version

    dataset = load_dataset(dataset_id, dataset_version, split='train')

    output_file = 'data/raw/ms_marco_5k.txt'  # Replace with your desired output file path

    extract_5k_lines(dataset, output_file)

if __name__ == "__main__":
    main()