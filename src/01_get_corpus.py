from config import *

from utils.data_utils import get_raw_dataset, get_clean_dataset, get_tokeniser_dictionaries, get_corpus

def main():
    dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dataset)
    vocab_to_int, _ = get_tokeniser_dictionaries(clean_dataset, min_frequency)
    get_corpus(clean_dataset, vocab_to_int, min_frequency)

if __name__ == "__main__":
    main()