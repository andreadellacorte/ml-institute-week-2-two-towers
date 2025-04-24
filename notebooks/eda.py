from config import *
from datasets import load_dataset
from utils.data_utils import get_raw_dataset

def main():
    get_raw_dataset(dataset_id, dataset_version, max_lines=5000)

if __name__ == "__main__":
    main()