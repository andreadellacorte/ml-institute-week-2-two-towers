import os
import re
import json
from datasets import load_dataset

dataset_id = "microsoft/ms_marco"  # Replace with your input file path
dataset_version = "v1.1"  # Replace with your desired dataset version
max_lines = 5000
min_frequency = None  # Set the minimum frequency for tokenization

def get_raw_dataset(dataset_id, dataset_version, max_lines=None):
    cache_file = f"data/raw/ms_marco_all.txt"  # Path to the cached dataset file

    if not os.path.exists(cache_file):
        print(f"Cache file not found. Downloading and saving dataset to {cache_file}...")
        dataset = load_dataset(dataset_id, dataset_version, split='train')
        with open(cache_file, "w", encoding='utf-8') as f:
            for item in dataset:
                json.dump(item, f)
                f.write("\n")
    else:
        print(f"Cache file found at {cache_file}. Using cached dataset.")

    # Proceed with loading the dataset from the cache file or using it directly
    with open(cache_file, "r") as f:
        dataset = [json.loads(line) for line in f]

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
            # Clean the dataset by removing unnecessary fields
            cleaned_item = {
                "query": re.sub('\W+',' ', item["query"]),
                "passages": [re.sub('\W+',' ', passage) for passage in item["passages"]["passage_text"]],
            }
            cleaned_data.append(cleaned_item)
        
        with open(cache_file, "w") as f:
            json.dump(cleaned_data, f, indent=4)
    else:
        print(f"Cache file found at {cache_file}. Using cached cleaned dataset.")

    with open(cache_file, "r") as f:
        cleaned_dataset = json.load(f)

    return cleaned_dataset

def clean_cache():
    # Clean the cache by removing the cached dataset files
    cache_files = [
        "data/processed/*.json"
    ]

    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Removed cache file: {cache_file}")
        else:
            print(f"Cache file not found: {cache_file}")

def get_tokenised_dataset(dataset, min_frequency=None):
    # Check if we already have a tokenized dataset with the same number of lines
    num_lines = len(dataset)
    
    cache_file = f"data/processed/ms_marco_tokenized_{num_lines}_lines_minfreq_{min_frequency if min_frequency else 'none'}.json"
    
    if os.path.exists(cache_file):
        print(f"Cache file found at {cache_file}. Using cached tokenized dataset.")
        with open(cache_file, "r") as f:
            tokenized_dataset = json.load(f)
        return tokenized_dataset

    print(f"Cache file not found. Tokenizing dataset and saving to {cache_file}...")
    
    word_frequency = {}

    for item in dataset:
        # Tokenize the query and passages (replace with actual tokenization logic)
        tokenized_query = item["query"].split()  # Replace with actual tokenization
        tokenized_passages = [passage.split() for passage in item["passages"]]  # Replace with actual tokenization

        # Update word frequency
        for word in tokenized_query:
            word_frequency[word] = word_frequency.get(word, 0) + 1
        for passage in tokenized_passages:
            for word in passage:
                word_frequency[word] = word_frequency.get(word, 0) + 1

    # Filter words based on min_frequency
    if min_frequency is not None:
        filtered_word_frequency = {word: freq for word, freq in word_frequency.items() if freq >= min_frequency}
        remaining_words = {word: idx for idx, word in enumerate(filtered_word_frequency.keys())}
        removed_words_count = len(word_frequency) - len(filtered_word_frequency)

        print(f"Tokenized dataset saved to {cache_file}.")

        print(f"Number of words in the original dataset: {len(word_frequency)}")
        print(f"Number of remaining words: {len(filtered_word_frequency)}")
        print(f"Number of removed words: {removed_words_count}")
        print(f"% of words removed: {removed_words_count / len(word_frequency) * 100:.2f}%")

        result = remaining_words
    else:
        words = {word: idx for idx, word in enumerate(word_frequency.keys())}

        print(f"Number of remaining words: {len(words)}")
        print(f"Number of removed words: 0")
    
        result = words

    # Add special tokens
    result["<UNK>"] = -1

    # Save the tokenized dataset to a cache file
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=4)
        
    print(f"Tokenized dataset saved to {cache_file}.")    

    return result

def main():
    dataset = get_raw_dataset(dataset_id, dataset_version, max_lines)
    clean_dataset = get_clean_dataset(dataset)
    get_tokenised_dataset(clean_dataset, min_frequency)

if __name__ == "__main__":
    main()